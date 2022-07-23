import asyncio
from aiohttp_client_cache import CachedSession, FileBackend
from itertools import product
import aiohttp
import pandas as pd
import TaskGraph as tg
from datetime import datetime, timedelta


# async def main():
#     concat = tg.MainNode(concat_dfs)
#     for date in [datetime(2022, 7, 12), datetime(2022, 7, 13), datetime(2022, 7, 14)]:
#         tg.SeedNode(date).async_task(get_game_ids).main_task(parse_game_json).add_child(concat)
#     concat.collect()
#     result = await concat.start()
#     await (await get_session()).close()
#     print(result)


class MultiIterator:

    def __init__(self, i=[], columns=[], dtypes={}):
        self.iterables = i
        self.df = pd.DataFrame(product(*i), columns=columns).astype(dtypes)
        self.clean_df()
        self.indices = self.df.itertuples(index=False, name=None)

    def clean_df(self):
        pass

    def __len__(self):
        return len(self.df)

    def __next__(self):
        return next(self.indices)

    def __iter__(self):
        return self


class WeekIterable(MultiIterator):

    def __init__(self, start_year, end_year, i=[], columns=[], truncate=True):
        weeks = self.create_weeks(start_year, end_year)
        self.truncate = truncate
        super().__init__(i=[*i, weeks], columns=[*columns, "start_date"])

    def clean_df(self):
        self.df["end_date"] = self.df["start_date"] + timedelta(days=6, hours=23, minutes=59, seconds=59)
        if self.truncate:
            self.df = self.df[self.df["start_date"] <= datetime.today()]

    @staticmethod
    def create_weeks(start_year, end_year):
        # Find the first Monday before Jan 1 or our start year.
        start = datetime(year=start_year, month=1, day=1).date()
        td = timedelta(days=1)
        while start.weekday() != 0:
            start -= td
        # Add the days to a list until we're after Dec 31st of end_year
        weeks = []
        week_td = timedelta(days=7)
        end = datetime(year=end_year, month=12, day=31).date()
        while start < end:
            weeks.append(pd.to_datetime(start))
            start += week_td
        return weeks

    def __next__(self):
        x = list(super().__next__())
        x[-2] = x[-2].to_pydatetime()
        x[-1] = x[-1].to_pydatetime()
        return tuple(x)


class GameData:

    def __init__(self, start_year, end_year):
        self.start_year = start_year
        self.end_year = end_year
        self.session = None
        self.urls = None
        self.tree = None
        self.result = None
        self.types = {"id": "Int64", "date": "datetime64", "state": "string", "type": "string",
                      "away_team_name": "string",
                      "away_team_id": "Int64", "away_team_score": "Int64", "home_team_name": "string",
                      "home_team_id": "Int64",
                      "home_team_score": "Int64", "venue_name": "string", "venue_id": "Int64",
                      "resume_date": "datetime64", "resume_from": "datetime64"}

    async def close(self):
        await self.session.close()
        await self.conn.close()
        # https://docs.aiohttp.org/en/stable/client_advanced.html - Graceful Shutdown
        await asyncio.sleep(.5)

    async def _init_http(self):
        itr = WeekIterable(start_year=self.start_year, end_year=self.end_year, truncate=True)
        cache_info = dict()
        today = datetime.today()
        for start, end in itr:
            url = "https://statsapi.mlb.com/api/v1/schedule?startDate={}&endDate={}&sportId=1".format(start.strftime('%m/%d/%Y'), end.strftime('%m/%d/%Y'))
            cache_info[url] = 30 #(today - end)
        self.urls = list(cache_info.keys())
        self.conn = aiohttp.TCPConnector(limit=5)
        self.session = CachedSession(cache=FileBackend(urls_expire_after=cache_info), conn=self.conn)
        await self.session.delete_expired_responses()

    async def init(self):
        await self._init_http()
        concat = tg.MainNode(self._concat)
        for url in self.urls:
            tg.SeedNode(url).async_task(self._http).main_task(self._parse).add_child(concat)
        self.tree = concat.main_task(self._clean).collect()

    async def _http(self, url):
        async with self.session.get(await url) as response:
            return await response.json(encoding="utf-8")

    def _parse(self, game_json):
        rows = []
        for date in game_json["dates"]:
            for game in date["games"]:
                row = [None] * len(self.types)
                row[0] = game["gamePk"]
                row[1] = game["gameDate"]
                row[2] = game["status"]["statusCode"]
                row[3] = game["gameType"]
                row[4] = game["teams"]["away"]["team"]["name"]
                row[5] = game["teams"]["away"]["team"]["id"]
                if "score" in game["teams"]["away"]:
                    row[6] = game["teams"]["away"]["score"]
                row[7] = game["teams"]["home"]["team"]["name"]
                row[8] = game["teams"]["home"]["team"]["id"]
                if "score" in game["teams"]["home"]:
                    row[9] = game["teams"]["home"]["score"]
                row[10] = game["venue"]["name"]
                row[11] = game["venue"]["id"]
                if "resumeDate" in game:
                    row[12] = game["resumeDate"]
                if "resumedFrom" in game:
                    row[13] = game["resumedFrom"]

                rows.append(row)
        return pd.DataFrame(rows, columns=self.types.keys()).astype(self.types)

    def _concat(self, *args):
        if len(args) == 0:
            return pd.DataFrame(columns=self.types.keys())
        return pd.concat(args)

    def _clean(self, dataframe: pd.DataFrame):
        return dataframe.drop_duplicates(subset=["id", "date", "state"]).sort_values(by=["id", "state"]).reset_index(
            drop=True)

    async def start(self):
        self.result = await self.tree.start()


async def main():
    gd = GameData(start_year=2022, end_year=2022)
    await gd.init()
    await gd.start()
    await gd.close()
    print(gd.result)


if __name__ == '__main__':
    asyncio.run(main())

    # asyncio.run(main())

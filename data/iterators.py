from datetime import timedelta, datetime
from itertools import product
from abc import abstractmethod
from collections.abc import Iterator
import pandas as pd

import data.constants as c
import data.utils as u


class IteratorResponse:

    def __init__(self, cache, cache_params, aux_params=None):
        self.cache = cache
        self.cache_params = cache_params
        # Used to pass params we don't want to consider in the cache. "verbose", "parallel", etc.
        if type(aux_params) is not dict:
            self.all_params = self.cache_params
        else:
            self.all_params = {**self.cache_params, **aux_params}


class BaseballIterator(Iterator):

    @abstractmethod
    def __next__(self) -> IteratorResponse:
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __iter__(self):
        return self


class SimpleIterator(BaseballIterator):

    def __init__(self, cache, params):
        self.iterator = iter([IteratorResponse(cache, params)])

    def __len__(self):
        return 1

    def __next__(self) -> IteratorResponse:
        return next(self.iterator)


class MultiIterator(BaseballIterator):

    def __init__(self, i=[], columns=[], dtypes={}):
        self.iterables = i
        self.df = pd.DataFrame(product(*i), columns=columns).astype(dtypes)
        self.clean_df()
        self.indices = self.df.itertuples(index=False, name=None)
        self.failure = []
        self.success = []

    def clean_df(self):
        pass

    def __len__(self):
        return len(self.df)

    def __next__(self):
        return next(self.indices)

    def add_failure(self, f):
        self.failure.append(f)

    def add_success(self, s):
        self.success.append(s)


class WeekIterable(MultiIterator):

    def __init__(self, i=[], columns=[], start_year=c.START_YEAR, end_year=c.END_YEAR):
        weeks = self.create_weeks(start_year, end_year)
        super().__init__(i=[*i, weeks], columns=[*columns, "start_date"])

    def clean_df(self):
        self.df["end_date"] = self.df["start_date"] + timedelta(days=6, hours=23, minutes=59, seconds=59)
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
        sunday_td = timedelta(days=6)
        week_td = timedelta(days=7)
        end = datetime(year=end_year, month=12, day=31).date()
        while start < end:
            if u.game_week(start, start + sunday_td):
                weeks.append(pd.to_datetime(start))
            start += week_td
        return weeks

    def __next__(self):
        x = list(super().__next__())
        x[-2] = x[-2].to_pydatetime().date()
        x[-1] = x[-1].to_pydatetime().date()
        return tuple(x)


class YearIterator(MultiIterator):

    def __init__(self, keys, start_year=c.START_YEAR, end_year=c.END_YEAR):
        super().__init__(i=[u.yl(start_year, end_year)], columns=["year"])
        self.keys = list(keys)

    def __next__(self):
        (year,) = super().__next__()
        ir = IteratorResponse(not u.today(year), dict([(key, year) for key in self.keys]))
        return ir


class StatcastIterator(WeekIterable):

    def __init__(self, teams=u.get_team_ids(), start_year=c.START_YEAR, end_year=c.END_YEAR):
        super().__init__(i=[teams], columns=["team"], start_year=start_year, end_year=end_year)

    def __next__(self):
        team, start, end = super().__next__()
        return IteratorResponse(not u.today(year=end.year, month=end.month, day=end.day),
                                {"start_dt": "{}-{}-{}".format(start.year, start.month, start.day),
                                 "end_dt": "{}-{}-{}".format(end.year, end.month, end.day), "team": team},
                                {"verbose": False, "parallel": False})


class StatcastPitcherIterator(WeekIterable):

    def __init__(self, start_year=c.START_YEAR, end_year=c.END_YEAR):
        self.pitchers = u.get_pitcher_df(start_year, end_year)
        super().__init__(i=[list(pd.unique(self.pitchers["player_id"]))], columns=["player_id"], start_year=start_year,
                         end_year=end_year)

    def clean_df(self):
        super().clean_df()
        og_columns = list(self.df.columns)
        self.df["year"] = self.df["start_date"].dt.year
        merged = pd.merge(self.pitchers, self.df, left_on=["player_id", "year"], right_on=["player_id", "year"])
        self.df = merged[og_columns]

    def __next__(self):
        player_id, start, end = super().__next__()
        return IteratorResponse(not u.today(year=end.year, month=end.month, day=end.day), {
            "start_dt": "{}-{}-{}".format(start.year, start.month, start.day),
            "end_dt": "{}-{}-{}".format(end.year, end.month, end.day),
            "player_id": player_id
        })


class PitchingStatsIterator(WeekIterable):

    def __init__(self, start_year=c.START_YEAR, end_year=c.END_YEAR):
        super().__init__(start_year=start_year, end_year=end_year)

    def __next__(self):
        start, end = super().__next__()
        return IteratorResponse(not u.today(year=end.year, month=end.month, day=end.day), {
            "start_dt": "{}-{}-{}".format(start.year, start.month, start.day),
            "end_dt": "{}-{}-{}".format(end.year, end.month, end.day),
        })


class TeamPitchingIterator(MultiIterator):

    def __init__(self, teams=u.get_team_ids(), start_year=c.START_YEAR, end_year=c.END_YEAR):
        super().__init__(i=[teams, u.yl(start_year, end_year)], columns=["team", "year"])

    def clean_df(self):
        self.df = self.df.drop(self.df[(self.df["team"] == "TBD") & (self.df["year"] > 2007)].index)
        self.df = self.df.drop(self.df[(self.df["team"] == "TBA") & (self.df["year"] < 2008)].index)
        self.df = self.df.drop(self.df[(self.df["team"] == "ANA") & (self.df["year"] > 2004)].index)
        self.df = self.df.drop(self.df[(self.df["team"] == "WSN") & (self.df["year"] < 2005)].index)
        self.df = self.df.drop(self.df[(self.df["team"] == "FLA") & (self.df["year"] > 2012)].index)

    def __next__(self):
        team, year = super().__next__()
        return IteratorResponse(not u.today(year), {
            "team": team,
            "start_season": year,
            "end_season": year
        })


class StatcastPitcherPitchMovementIterator(MultiIterator):

    def __init__(self, pitches=c.PITCH_TYPES, start_year=c.START_YEAR, end_year=c.END_YEAR):
        super().__init__(i=[pitches, u.yl(start_year, end_year)], columns=["pitch", "year"])

    def __next__(self):
        pitch, year = super().__next__()
        return IteratorResponse(not u.today(year), {
            "pitch_type": pitch,
            "year": year,
        })

import data.constants as c
import hashlib
import json
import pickle
from copy import copy
from datetime import datetime, timedelta
from os.path import join, exists
import numpy as np
import pandas as pd
import requests
from pybaseball import fangraphs_teams, bwar_pitch
from pybaseball.statcast_pitcher import statcast_pitcher_arsenal_stats as sp_arsenal_stats
from pybaseball.statcast_pitcher import statcast_pitcher_active_spin as sp_active_spin


def get_team_ids():
    try:
        get_team_ids.TEAM_IDS
    except AttributeError:
        # Not sure if all APIs take the teamID or franchID, so adding both to a set.
        team_df = pd.concat([fangraphs_teams(c.START_YEAR), fangraphs_teams(c.END_YEAR)])
        team_ids = set(pd.unique(team_df["teamID"]))
        team_ids.update(pd.unique(team_df["franchID"]))
        get_team_ids.TEAM_IDS = team_ids
    return get_team_ids.TEAM_IDS


def get_pitcher_df(start_year, end_year):
    p = bwar_pitch(return_all=True)[["mlb_ID", "year_ID", "IPouts"]].astype(
        {"mlb_ID": pd.Int64Dtype(), "year_ID": pd.Int64Dtype()}).rename(
        columns={"year_ID": "year", "mlb_ID": "player_id"})
    pitchers = p[(p.year >= start_year) & (p.year <= end_year)]
    pitchers = pitchers[pitchers["IPouts"] > 0]
    pitchers = pitchers.drop_duplicates(subset=["year", "player_id"])
    return pitchers


def get_valid_days():
    try:
        get_valid_days.VALID_DAYS
    except AttributeError:
        get_valid_days.VALID_DAYS = None

    path = join("data", "cache", "valid_days.pkl")
    if get_valid_days.VALID_DAYS is None and exists(path):
        with open(path, "rb") as f:
            get_valid_days.VALID_DAYS = pickle.load(f)
    elif get_valid_days.VALID_DAYS is None:
        get_valid_days.VALID_DAYS = create_valid_days()
        with open(path, "wb") as f:
            pickle.dump(get_valid_days.VALID_DAYS, f)
    return get_valid_days.VALID_DAYS


def create_valid_days():
    dates = set()
    for year in range(c.START_YEAR, c.END_YEAR + 1):
        url = "https://statsapi.mlb.com/api/v1/schedule?startDate=01/01/{}&endDate=12/31/{}&sportId=1".format(year, year)
        r = requests.get(url)
        for d in r.json()["dates"]:
            if type(d) is dict and "date" in d and "games" in d and type(d["games"]) is list:
                real_games = any([x == "R" for x in [g["gameType"] for g in d["games"]]])
                if real_games:
                    s_date = d["date"]
                    p_datetime = datetime.strptime(s_date, "%Y-%m-%d")
                    p_date = p_datetime.date()
                    dates.add(p_date)
    return dates


def today(year=None, month=None, day=None):
    if year is None:
        return False

    now = datetime.now()
    if year > now.year:
        return True
    elif year == now.year and (month is None or month > now.month):
        return True
    elif year == now.year and month == now.month and (day is None or day >= now.day):
        return True

    return False


def game_date(year, month=None, day=None):
    try:
        d = datetime(year=year, month=month, day=day).date()
        return d in get_valid_days()
    except ValueError:
        return False


def game_week(start_day, end_day):
    td = timedelta(1)
    d = copy(start_day)
    while d <= end_day:
        if d in get_valid_days():
            return True
        d += td
    return False


def yl(start_year=c.START_YEAR, end_year=c.END_YEAR):
    return list(range(start_year, min(c.CURRENT_YEAR + 1, end_year + 1)))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# We're using the dict returns od the iterators below to identify the calls. Using MD5 to give us an idempotent id.
def dmd5(d: dict):
    return hashlib.md5(json.dumps(d, cls=NpEncoder).encode()).hexdigest()


def chadwick_cleanup(df):
    return df.dropna(subset=['mlb_played_first', 'mlb_played_last']).astype({"mlb_played_first": pd.Int64Dtype(), "mlb_played_first": pd.Int64Dtype()})


def copy_year(column):
    def inner_copy_year(df):
        ndf = df.dropna(subset=[column])
        ndf["year"] = ndf[column].astype(pd.Int64Dtype())
        return ndf
    return inner_copy_year


def noop_clean(df):
    return df


# These methods are silly and don't include a year column when they should
def statcast_pitcher_active_spin(year: int, minP: int = 250, _type: str = 'spin-based') -> pd.DataFrame:
    data = sp_active_spin(year=year, minP=minP, _type=_type)
    data["year"] = year
    return data


def statcast_pitcher_arsenal_stats(year: int, minPA: int = 25) -> pd.DataFrame:
    data = sp_arsenal_stats(year=year, minPA=minPA)
    data["year"] = year
    return data


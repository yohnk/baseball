import datetime
import hashlib
import json
from copy import copy
import pandas as pd
import numpy as np
from pybaseball import bwar_pitch, chadwick_register, fangraphs_teams, pitching, pitching_post, pitching_stats, \
    pitching_stats_bref, pitching_stats_range, player_search_list, playerid_lookup, playerid_reverse_lookup, statcast, \
    statcast_pitcher, team_pitching_bref, cache
from pybaseball.statcast_pitcher import statcast_pitcher_exitvelo_barrels, statcast_pitcher_expected_stats, \
    statcast_pitcher_pitch_arsenal, statcast_pitcher_arsenal_stats, statcast_pitcher_pitch_movement, \
    statcast_pitcher_active_spin, statcast_pitcher_percentile_ranks, statcast_pitcher_spin_dir_comp

cache.enable()

START_YEAR = 2000
END_YEAR = datetime.date.today().year

chadwick = chadwick_register()

# Grab all players whos last year is during or after our start_year
player_ids = set(pd.unique(chadwick[chadwick["mlb_played_last"].gt(START_YEAR - 1)]["key_mlbam"]))

# Not sure if all APIs take the teamID or franchID, so adding both to a set. If it's not used the API call should fail.
team_df = pd.concat([fangraphs_teams(START_YEAR), fangraphs_teams(END_YEAR)])
team_ids = set(pd.unique(team_df["teamID"]))
team_ids.update(pd.unique(team_df["franchID"]))

# pitch_types = ['SL', 'CH', 'FC', 'FF', 'FS']
pitch_types = ["FF", "SIFT", "CH", "CUKC", "FC", "SL", "FS"]

# We're using the dict returns od the iterators below to identify the calls. Using MD5 to give us an idempotent id.
def dmd5(d: dict):
    return hashlib.md5(json.dumps(d).encode()).hexdigest()


# Iterators that are used to create multiple API requests.
class YearIterator:

    def __init__(self, keys, start=START_YEAR, stop=END_YEAR):
        self.start = start
        self.stop = stop
        self.i = start
        self.keys = copy(keys)

    def __iter__(self):
        return self

    def __next__(self):
        year = copy(self.i)
        self.i += 1
        if year <= self.stop:
            d = dict()
            for k in self.keys:
                d[k] = year
            return d
        else:
            raise StopIteration


class StatcastIterator:

    def __init__(self, start=START_YEAR, stop=END_YEAR, teams=team_ids):
        self.teams = iter(teams)
        self.current_team = next(self.teams)
        self.start = start
        self.stop = stop
        self.years = iter(list(range(self.start, self.stop + 1)))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            year = next(self.years)
        except StopIteration:
            self.current_team = next(self.teams)
            self.years = iter(list(range(START_YEAR, END_YEAR + 1)))
            year = next(self.years)
        return {
            "start_dt": "{}-1-1".format(year),
            "end_dt": "{}-12-31".format(year),
            "team": self.current_team,
            "verbose": False,
            "parallel": True
        }


class StatcastPitcherIterator:

    def __init__(self, ids=player_ids, start=START_YEAR, end=END_YEAR):
        self.player_ids = iter(ids)
        self.start = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        pid = int(next(self.player_ids))
        return {
            "start_dt": "{}-1-1".format(self.start),
            "end_dt": "{}-12-31".format(self.end),
            "player_id": pid
        }


class PitchingStatsIterator:

    def __init__(self, start=START_YEAR, stop=END_YEAR):
        self.start = start
        self.stop = stop
        self.i = start

    def __iter__(self):
        return self

    def __next__(self):
        year = copy(self.i)
        self.i += 1
        if year <= self.stop:
            return {
                "start_dt": "{}-1-1".format(year),
                "end_dt": "{}-12-31".format(year)
            }
        else:
            raise StopIteration


class TeamPitchingIterator:

    def __init__(self, teams=team_ids):
        self.teams = iter(teams)
        self.current_team = next(self.teams)
        self.years = iter(list(range(START_YEAR, END_YEAR + 1)))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            year = next(self.years)
        except StopIteration:
            self.current_team = next(self.teams)
            self.years = iter(list(range(START_YEAR, END_YEAR + 1)))
            year = next(self.years)

        return {
            "team": self.current_team,
            "start_season": year,
            "end_season": year
        }


class StatcastPitcherPitchMovementIterator:

    def __init__(self, pitches=pitch_types):
        self.pitches = iter(pitches)
        self.current_pitch = next(self.pitches)
        self.years = iter(list(range(START_YEAR, END_YEAR + 1)))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            year = next(self.years)
        except StopIteration:
            self.current_pitch = next(self.pitches)
            self.years = iter(list(range(START_YEAR, END_YEAR + 1)))
            year = next(self.years)

        return {
            "pitch_type": self.current_pitch,
            "year": year,
        }


# A list of API methods and the iterable dicts of parameters used to "span" the API.
api_methods = {
    playerid_reverse_lookup: [{
        "player_ids": [477132],
        "key_type": "mlbam"
    }],
    player_search_list: [{
        "player_list": [("kershaw", "clayton")]
    }],
    playerid_lookup: [{
        "last": "kershaw",
        "first": "clayton",
        "fuzzy": False
    }],
    chadwick_register: [{
        "save": True
    }],
    fangraphs_teams: [{
        "season": None,
        "league": "ALL"
    }],
    statcast: StatcastIterator(),
    statcast_pitcher: StatcastPitcherIterator(),
    pitching_stats_bref: YearIterator(keys=["season"], start=2008),
    pitching_stats_range: PitchingStatsIterator(start=2008),
    bwar_pitch: [{
        "return_all": True
    }],
    pitching_stats: YearIterator(keys=["start_season", "end_season"]),
    team_pitching_bref: TeamPitchingIterator(),
    pitching: [{}],
    pitching_post: [{}],
    statcast_pitcher_exitvelo_barrels: YearIterator(keys=["year"]),
    statcast_pitcher_expected_stats: YearIterator(keys=["year"]),
    statcast_pitcher_pitch_arsenal: YearIterator(keys=["year"]),
    statcast_pitcher_arsenal_stats: YearIterator(keys=["year"]),
    statcast_pitcher_pitch_movement: StatcastPitcherPitchMovementIterator(),
    statcast_pitcher_active_spin: YearIterator(keys=["year"]),
    statcast_pitcher_percentile_ranks: YearIterator(keys=["year"]),
    statcast_pitcher_spin_dir_comp: YearIterator(keys=["year"])
}


def copy_year(column):
    def inner_copy_year(df):
        ndf = df.dropna(subset=[column])
        ndf["year"] = ndf[column].astype(pd.Int64Dtype())
        return ndf
    return inner_copy_year


def chadwick_cleanup(df):
    return df.dropna(subset=['mlb_played_first', 'mlb_played_last']).astype({"mlb_played_first": pd.Int64Dtype(), "mlb_played_first": pd.Int64Dtype()})


# We want to introduce a consistent year column. Also a hook for any other cleanup.
cleanup_methods = {
    playerid_reverse_lookup: chadwick_cleanup,
    player_search_list: chadwick_cleanup,
    playerid_lookup: chadwick_cleanup,
    chadwick_register: chadwick_cleanup,
    fangraphs_teams: copy_year("yearID"),
    statcast: copy_year("game_year"),
    statcast_pitcher: copy_year("game_year"),
    bwar_pitch: copy_year("year_ID"),
    pitching_stats: copy_year("Season"),
    team_pitching_bref: copy_year("Year"),
    pitching: copy_year("yearID"),
    pitching_post: copy_year("yearID"),
    statcast_pitcher_expected_stats: copy_year("year"),
    statcast_pitcher_pitch_movement: copy_year("year"),
    statcast_pitcher_percentile_ranks: copy_year("year"),
    statcast_pitcher_spin_dir_comp: copy_year("year")
}

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import data.constants as c
import tqdm
import data.utils as u
from abc import ABC
from data.cache_provider import FileCache

from pybaseball import bwar_pitch, chadwick_register, fangraphs_teams, pitching_stats, \
    pitching_stats_bref, pitching_stats_range, player_search_list, playerid_lookup, playerid_reverse_lookup, statcast, \
    statcast_pitcher, team_pitching_bref
from pybaseball.statcast_pitcher import statcast_pitcher_exitvelo_barrels, statcast_pitcher_expected_stats, \
    statcast_pitcher_pitch_arsenal, statcast_pitcher_pitch_movement, \
    statcast_pitcher_percentile_ranks, statcast_pitcher_spin_dir_comp


from iterators import BaseballIterator, StatcastIterator, StatcastPitcherIterator, YearIterator, PitchingStatsIterator, \
    TeamPitchingIterator, StatcastPitcherPitchMovementIterator, SimpleIterator


class Supplier(ABC):

    def __init__(self, api_method, iterator: BaseballIterator, cleanup=u.noop_clean):
        self.api_method = api_method
        self.iterator = iterator
        self.cleanup = cleanup
        self.cache = FileCache(base_name=self.name())
        self.api_response = []

        # For stats
        self.cache_reads = []
        self.api_reads = []
        self.api_failure = []

    def __len__(self):
        return len(self.iterator)

    def name(self):
        return self.api_method.__qualname__

    def _api(self):
        print(c.NUM_THREADS)
        with ThreadPoolExecutor(max_workers=c.NUM_THREADS) as executor, tqdm.tqdm(total=len(self.iterator), desc=self.name() + " API", disable_override=False) as progress:
            futures = [executor.submit(self._api_tick, ir) for ir in self.iterator]
            for _ in as_completed(futures):
                progress.update()
            # for ir in self.iterator:
            #     self._api_tick(ir)
            #     progress.update()

    def _merge(self):
        with tqdm.tqdm(total=1, desc=self.name() + " Merge", disable_override=False) as progress:
            res = pd.concat(self.api_response).reset_index()
            progress.update()
            return res

    def run(self):
        self._api()
        r = self._merge()
        with tqdm.tqdm(total=1, desc=self.name() + " Cleanup", disable_override=False) as progress:
            r = self.cleanup(r)
            progress.update()
            return r

    def _api_tick(self, ir):
        if ir.cache and self.cache.exists(**ir.cache_params):
            self.cache_reads.append(ir)
            r = self.cache.get(**ir.cache_params)
            self.api_response.append(r)
        else:
            try:
                r = self.api_method(**ir.all_params)
                self.api_reads.append(ir)
                self.api_response.append(r)
                if ir.cache:
                    self.cache.store(r, **ir.cache_params)
            except Exception as e:
                self.api_failure.append((ir, e))


# class PlayerReverseSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(playerid_reverse_lookup, SimpleIterator(True, {"player_ids": [477132], "key_type": "mlbam"}),
#                          u.chadwick_cleanup)
#
#
# class PlayerListSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(player_search_list, SimpleIterator(True, {"player_list": [("kershaw", "clayton")]})
#                          , u.chadwick_cleanup)
#
#
# class PlayerIDSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(playerid_lookup, SimpleIterator(True, {"last": "kershaw", "first": "clayton", "fuzzy": False}),
#                          u.chadwick_cleanup)
#
#
# class ChadwickSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(chadwick_register, SimpleIterator(False, {"save": True}), u.chadwick_cleanup)
#
#
# class FangraphsTeamsSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(fangraphs_teams, SimpleIterator(False, {"season": None, "league": "ALL"}), u.copy_year("yearID"))
#
#
class StatcastSupplier(Supplier):

    def __init__(self):
        super().__init__(statcast, StatcastIterator(), u.copy_year("game_year"))


# class StatcastPitcherSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(statcast_pitcher, StatcastPitcherIterator(), u.copy_year("game_year"))
#
#
# class PitchingBrefSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(pitching_stats_bref, YearIterator(keys=["season"], start_year=2008))
#
#
# class PitchingRangeSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(pitching_stats_range, PitchingStatsIterator(start_year=2008))
#
#
# class BWarSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(bwar_pitch, SimpleIterator(False, {"return_all": True}), u.copy_year("year_ID"))
#
#
# class PitchingStatsSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(pitching_stats, YearIterator(keys=["start_season", "end_season"]), u.copy_year("Season"))
#
#
# class TeamPitchingSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(team_pitching_bref, TeamPitchingIterator(), u.copy_year("Year"))
#
#
# class SCExitVeloSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(statcast_pitcher_exitvelo_barrels, YearIterator(keys=["year"]))
#
#
# class SCExpectedStatsSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(statcast_pitcher_expected_stats, YearIterator(keys=["year"]), u.copy_year("year"))
#
#
# class SCPitchArsenalSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(statcast_pitcher_pitch_arsenal, YearIterator(keys=["year"]))
#
#
# class SCArsenalStatsSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(u.statcast_pitcher_arsenal_stats, YearIterator(keys=["year"]))
#
#
# class SCPitchMovementSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(statcast_pitcher_pitch_movement, StatcastPitcherPitchMovementIterator(), u.copy_year("year"))
#
#
# class SCActiveSpinSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(u.statcast_pitcher_active_spin, YearIterator(keys=["year"]))
#
#
# class SCPercentileRanksSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(statcast_pitcher_percentile_ranks, YearIterator(keys=["year"]), u.copy_year("year"))
#
#
# class SCSpinDirSupplier(Supplier):
#
#     def __init__(self):
#         super().__init__(statcast_pitcher_spin_dir_comp, YearIterator(keys=["year"]), u.copy_year("year"))

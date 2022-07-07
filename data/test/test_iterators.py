import pickle

from data.boilerplate import YearIterator, StatcastIterator, StatcastPitcherIterator, PitchingStatsIterator, \
    TeamPitchingIterator, StatcastPitcherPitchMovementIterator
import unittest


class TestIterators(unittest.TestCase):

    def validate_iterator(self, i_test, i_real):
        for cache_test, x_test in i_test:
            cache_real, x_real = next(i_real)
            self.assertEqual(cache_real, cache_test)
            self.assertEqual(x_real, x_test)

        try:
            next(i_test)
            self.fail()
        except StopIteration:
            pass

    def test_year_iterator(self):
        results = iter([
            (False, {'test1': 2015, 'test2': 2015}),
            (False, {'test1': 2016, 'test2': 2016}),
            (False, {'test1': 2017, 'test2': 2017}),
            (False, {'test1': 2018, 'test2': 2018}),
            (False, {'test1': 2019, 'test2': 2019}),
            (False, {'test1': 2020, 'test2': 2020}),
            (False, {'test1': 2021, 'test2': 2021}),
            (True, {'test1': 2022, 'test2': 2022})
        ])
        yi = YearIterator(keys=["test1", "test2"], start_year=2015, end_year=2025)
        self.validate_iterator(yi, results)

    def test_statcast_iterator(self):
        si = StatcastIterator(teams=["A", "B"], start_year=2020, end_year=2021)
        with open("statcast_iterator_result.pkl", "rb") as f:
            results = pickle.load(f)
        self.validate_iterator(si, iter(results))

    def test_statcast_pitcher_iterator(self):
        spi = StatcastPitcherIterator(ids=[1, 2], start_year=2020, end_year=2021)
        with open("statcast_pitcher_iterator_result.pkl", "rb") as f:
            results = pickle.load(f)
        self.validate_iterator(spi, iter(results))

    def test_pitching_stats_iterator(self):
        psi = PitchingStatsIterator(start_year=1940, end_year=1950)
        with open("pitching_stats_iterator_result.pkl", "rb") as f:
            results = pickle.load(f)
        self.validate_iterator(psi, iter(results))

    def test_team_pitching_iterator(self):
        tpi = TeamPitchingIterator(teams=["A", "B"], start_year=2020, end_year=2021)
        results = iter([
            (False, {'team': 'A', 'start_season': 2020, 'end_season': 2020}),
            (False, {'team': 'A', 'start_season': 2021, 'end_season': 2021}),
            (False, {'team': 'B', 'start_season': 2020, 'end_season': 2020}),
            (False, {'team': 'B', 'start_season': 2021, 'end_season': 2021})
        ])
        self.validate_iterator(tpi, results)

    def test_statcast_pitcher_pitch_movement_iterator(self):
        sppmi = StatcastPitcherPitchMovementIterator(pitches=["SL", "FF"], start_year=1950, end_year=1955)
        results = iter([
            (False, {'pitch_type': 'SL', 'year': 1950}),
            (False, {'pitch_type': 'SL', 'year': 1951}),
            (False, {'pitch_type': 'SL', 'year': 1952}),
            (False, {'pitch_type': 'SL', 'year': 1953}),
            (False, {'pitch_type': 'SL', 'year': 1954}),
            (False, {'pitch_type': 'SL', 'year': 1955}),
            (False, {'pitch_type': 'FF', 'year': 1950}),
            (False, {'pitch_type': 'FF', 'year': 1951}),
            (False, {'pitch_type': 'FF', 'year': 1952}),
            (False, {'pitch_type': 'FF', 'year': 1953}),
            (False, {'pitch_type': 'FF', 'year': 1954}),
            (False, {'pitch_type': 'FF', 'year': 1955})
        ])
        self.validate_iterator(sppmi, results)


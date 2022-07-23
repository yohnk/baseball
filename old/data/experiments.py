import io
from numpy import mean
import pandas as pd
import requests

# A class to run experiments and document results
laa_url = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&team=LAA&game_date_gt=2022-4-10&game_date_lt=2022-4-10&type=details"
hou_url = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&team=HOU&game_date_gt=2022-4-10&game_date_lt=2022-4-10&type=details"

game_url = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&game_pk=661039&type=details"
game_all_stats_url = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&game_pk=661039&type=details&chk_stats_pa=on&chk_stats_abs=on&chk_stats_bip=on&chk_stats_hits=on&chk_stats_singles=on&chk_stats_dbls=on&chk_stats_triples=on&chk_stats_hrs=on&chk_stats_so=on&chk_stats_k_percent=on&chk_stats_bb=on&chk_stats_bb_percent=on&chk_stats_babip=on&chk_stats_iso=on&chk_stats_run_exp=on&chk_stats_ba=on&chk_stats_xba=on&chk_stats_xbadiff=on&chk_stats_slg=on&chk_stats_xslg=on&chk_stats_xslgdiff=on&chk_stats_obp=on&chk_stats_xobp=on&chk_stats_woba=on&chk_stats_xwoba=on&chk_stats_wobadiff=on&chk_stats_velocity=on&chk_stats_launch_speed=on&chk_stats_launch_angle=on&chk_stats_bbdist=on&chk_stats_spin_rate=on&chk_stats_plate_x=on&chk_stats_plate_z=on&chk_stats_release_pos_x=on&chk_stats_release_pos_z=on&chk_stats_pos3_int_start_distance=on&chk_stats_pos4_int_start_distance=on&chk_stats_pos6_int_start_distance=on&chk_stats_pos5_int_start_distance=on&chk_stats_pos7_int_start_distance=on&chk_stats_pos8_int_start_distance=on&chk_stats_pos9_int_start_distance=on&chk_stats_effective_speed=on&chk_stats_release_extension=on"


def df_from_url(url, sort=[]):
    return pd.read_csv(io.StringIO(requests.get(url, timeout=None).content.decode('utf-8'))).sort_values(sort)


def check_by_row(df1, df2):
    all_good = len(df1) == len(df2)
    try:
        team_itr = df1.iterrows()
        game_itr = df2.iterrows()
        while all_good:
            _, team = next(team_itr)
            _, game = next(game_itr)
            if not team.equals(game):
                all_good = False
                print(team.eq(game))
    except StopIteration:
        pass

    return all_good


def diff_between_game_team():
    # What's the difference between requests by game id or asking for a team & day
    # Conclusion - if you make a request with the date and BOTH teams, you get the same as if you requested by game_id

    laa_pd = df_from_url(laa_url)
    hou_pd = df_from_url(hou_url)
    team_pd = pd.concat([laa_pd, hou_pd]).sort_values(['release_speed', 'release_pos_x', 'release_pos_z'])

    game_pd = df_from_url(game_url, sort=['release_speed', 'release_pos_x', 'release_pos_z'])

    print("Team Len", len(team_pd))
    print("Game Len", len(game_pd))
    print("DF Equal", team_pd.equals(game_pd))
    print("Col Equal", all(team_pd.columns == game_pd.columns))
    print("Row Equal", check_by_row(team_pd, game_pd))


def time_game_vs_team():
    # Is it faster to request by game_id or for both teams?
    # Result - asking by game_id is ~half of asking for both teams.
    laa_times = []
    hou_times = []
    game_times = []
    for _ in range(5):
        laa_times.append(requests.get(laa_url, timeout=None).elapsed)
        hou_times.append(requests.get(hou_url, timeout=None).elapsed)
        game_times.append(requests.get(game_url, timeout=None).elapsed)

    print("Team Elapsed", mean(laa_times), mean(hou_times), mean(laa_times) + mean(hou_times))
    print("Game Elapsed", mean(game_times))


def simple_vs_extra_params():
    # Does adding the extra URL "stat" parameters change the default behavior?
    # Result - Nope, dead equal
    simple = df_from_url(game_url, sort=['release_speed', 'release_pos_x', 'release_pos_z'])
    extra = df_from_url(game_all_stats_url, sort=['release_speed', 'release_pos_x', 'release_pos_z'])

    print("simple Len", len(simple))
    print("extra Len", len(extra))
    print("DF Equal", simple.equals(extra))
    print("Col Equal", all(simple.columns == extra.columns))
    print("Row Equal", check_by_row(simple, extra))


if __name__ == "__main__":
    simple_vs_extra_params()

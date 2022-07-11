from datetime import datetime, date, timedelta
import hashlib
import json
import pickle
from copy import copy
from os.path import join, exists

import numpy as np
import pandas as pd
from itertools import product

import requests
from pybaseball import bwar_pitch, chadwick_register, fangraphs_teams, pitching, pitching_post, pitching_stats, \
    pitching_stats_bref, pitching_stats_range, player_search_list, playerid_lookup, playerid_reverse_lookup, statcast, \
    statcast_pitcher, team_pitching_bref, cache
from pybaseball.statcast_pitcher import statcast_pitcher_exitvelo_barrels, statcast_pitcher_expected_stats, \
    statcast_pitcher_pitch_arsenal, statcast_pitcher_pitch_movement, \
    statcast_pitcher_percentile_ranks, statcast_pitcher_spin_dir_comp

from pybaseball.statcast_pitcher import statcast_pitcher_arsenal_stats as sp_arsenal_stats
from pybaseball.statcast_pitcher import statcast_pitcher_active_spin as sp_active_spin

cache.enable()

START_YEAR = 2000
END_YEAR = date.today().year
CURRENT_YEAR = date.today().year
CURRENT_MONTH = date.today().month

chadwick = chadwick_register()

# Grab all players whos last year is during or after our start_year
# player_ids = set(pd.unique(chadwick[chadwick["mlb_played_last"].gt(START_YEAR - 1)]["key_mlbam"]))

# Not sure if all APIs take the teamID or franchID, so adding both to a set. If it's not used the API call should fail.
# Update - removing the teamID since it adds a lot of unnecessary calls
team_df = pd.concat([fangraphs_teams(START_YEAR), fangraphs_teams(END_YEAR)])
team_ids = set(pd.unique(team_df["franchID"]))

# pitch_types = ['SL', 'CH', 'FC', 'FF', 'FS']
pitch_types = ["FF", "SIFT", "CH", "CUKC", "FC", "SL", "FS"]
months = list(range(1, 13))
days = list(range(1, 32))

# Manually went through the response and picked the best type
data_types = {
    'FangraphsDataTable.fetch': {'IDfg': 'Int64', 'Season': 'Int64', 'Name': 'string', 'Team': 'string', 'Age': 'Int64',
                                 'W': 'Int64', 'L': 'Int64', 'WAR': 'float64', 'ERA': 'float64', 'G': 'Int64',
                                 'GS': 'Int64', 'CG': 'Int64', 'ShO': 'Int64', 'SV': 'Int64', 'BS': 'float64',
                                 'IP': 'float64', 'TBF': 'Int64', 'H': 'Int64', 'R': 'Int64', 'ER': 'Int64',
                                 'HR': 'Int64', 'BB': 'Int64', 'IBB': 'Int64', 'HBP': 'Int64', 'WP': 'Int64',
                                 'BK': 'Int64', 'SO': 'Int64', 'GB': 'float64', 'FB': 'float64', 'LD': 'float64',
                                 'IFFB': 'float64', 'Balls': 'float64', 'Strikes': 'float64', 'Pitches': 'float64',
                                 'RS': 'float64', 'IFH': 'float64', 'BU': 'float64', 'BUH': 'float64', 'K/9': 'float64',
                                 'BB/9': 'float64', 'K/BB': 'float64', 'H/9': 'float64', 'HR/9': 'float64',
                                 'AVG': 'float64', 'WHIP': 'float64', 'BABIP': 'float64', 'LOB%': 'float64',
                                 'FIP': 'float64', 'GB/FB': 'float64', 'LD%': 'float64', 'GB%': 'float64',
                                 'FB%': 'float64', 'IFFB%': 'float64', 'HR/FB': 'float64', 'IFH%': 'float64',
                                 'BUH%': 'float64', 'Starting': 'float64', 'Start-IP': 'float64',
                                 'Relieving': 'float64', 'Relief-IP': 'float64', 'RAR': 'float64', 'Dollars': 'string',
                                 'tERA': 'float64', 'xFIP': 'float64', 'WPA': 'float64', '-WPA': 'float64',
                                 '+WPA': 'float64', 'RE24': 'float64', 'REW': 'float64', 'pLI': 'float64',
                                 'inLI': 'float64', 'gmLI': 'float64', 'exLI': 'float64', 'Pulls': 'Int64',
                                 'WPA/LI': 'float64', 'Clutch': 'float64', 'FB% 2': 'float64', 'FBv': 'float64',
                                 'SL%': 'float64', 'SLv': 'float64', 'CT%': 'float64', 'CTv': 'float64',
                                 'CB%': 'float64', 'CBv': 'float64', 'CH%': 'float64', 'CHv': 'float64',
                                 'SF%': 'float64', 'SFv': 'float64', 'KN%': 'float64', 'KNv': 'float64',
                                 'XX%': 'float64', 'PO%': 'float64', 'wFB': 'float64', 'wSL': 'float64',
                                 'wCT': 'float64', 'wCB': 'float64', 'wCH': 'float64', 'wSF': 'float64',
                                 'wKN': 'float64', 'wFB/C': 'float64', 'wSL/C': 'float64', 'wCT/C': 'float64',
                                 'wCB/C': 'float64', 'wCH/C': 'float64', 'wSF/C': 'float64', 'wKN/C': 'float64',
                                 'O-Swing%': 'float64', 'Z-Swing%': 'float64', 'Swing%': 'float64',
                                 'O-Contact%': 'float64', 'Z-Contact%': 'float64', 'Contact%': 'float64',
                                 'Zone%': 'float64', 'F-Strike%': 'float64', 'SwStr%': 'float64', 'HLD': 'float64',
                                 'SD': 'Int64', 'MD': 'Int64', 'ERA-': 'Int64', 'FIP-': 'Int64', 'xFIP-': 'float64',
                                 'K%': 'float64', 'BB%': 'float64', 'SIERA': 'float64', 'RS/9': 'float64',
                                 'E-F': 'float64', 'FA% (sc)': 'float64', 'FT% (sc)': 'float64', 'FC% (sc)': 'float64',
                                 'FS% (sc)': 'float64', 'FO% (sc)': 'float64', 'SI% (sc)': 'float64',
                                 'SL% (sc)': 'float64', 'CU% (sc)': 'float64', 'KC% (sc)': 'float64',
                                 'EP% (sc)': 'float64', 'CH% (sc)': 'float64', 'SC% (sc)': 'float64',
                                 'KN% (sc)': 'float64', 'UN% (sc)': 'float64', 'vFA (sc)': 'float64',
                                 'vFT (sc)': 'float64', 'vFC (sc)': 'float64', 'vFS (sc)': 'float64',
                                 'vFO (sc)': 'float64', 'vSI (sc)': 'float64', 'vSL (sc)': 'float64',
                                 'vCU (sc)': 'float64', 'vKC (sc)': 'float64', 'vEP (sc)': 'float64',
                                 'vCH (sc)': 'float64', 'vSC (sc)': 'float64', 'vKN (sc)': 'float64',
                                 'FA-X (sc)': 'float64', 'FT-X (sc)': 'float64', 'FC-X (sc)': 'float64',
                                 'FS-X (sc)': 'float64', 'FO-X (sc)': 'float64', 'SI-X (sc)': 'float64',
                                 'SL-X (sc)': 'float64', 'CU-X (sc)': 'float64', 'KC-X (sc)': 'float64',
                                 'EP-X (sc)': 'float64', 'CH-X (sc)': 'float64', 'SC-X (sc)': 'float64',
                                 'KN-X (sc)': 'float64', 'FA-Z (sc)': 'float64', 'FT-Z (sc)': 'float64',
                                 'FC-Z (sc)': 'float64', 'FS-Z (sc)': 'float64', 'FO-Z (sc)': 'float64',
                                 'SI-Z (sc)': 'float64', 'SL-Z (sc)': 'float64', 'CU-Z (sc)': 'float64',
                                 'KC-Z (sc)': 'float64', 'EP-Z (sc)': 'float64', 'CH-Z (sc)': 'float64',
                                 'SC-Z (sc)': 'float64', 'KN-Z (sc)': 'float64', 'wFA (sc)': 'float64',
                                 'wFT (sc)': 'float64', 'wFC (sc)': 'float64', 'wFS (sc)': 'float64',
                                 'wFO (sc)': 'float64', 'wSI (sc)': 'float64', 'wSL (sc)': 'float64',
                                 'wCU (sc)': 'float64', 'wKC (sc)': 'float64', 'wEP (sc)': 'float64',
                                 'wCH (sc)': 'float64', 'wSC (sc)': 'float64', 'wKN (sc)': 'float64',
                                 'wFA/C (sc)': 'float64', 'wFT/C (sc)': 'float64', 'wFC/C (sc)': 'float64',
                                 'wFS/C (sc)': 'float64', 'wFO/C (sc)': 'float64', 'wSI/C (sc)': 'float64',
                                 'wSL/C (sc)': 'float64', 'wCU/C (sc)': 'float64', 'wKC/C (sc)': 'float64',
                                 'wEP/C (sc)': 'float64', 'wCH/C (sc)': 'float64', 'wSC/C (sc)': 'float64',
                                 'wKN/C (sc)': 'float64', 'O-Swing% (sc)': 'float64', 'Z-Swing% (sc)': 'float64',
                                 'Swing% (sc)': 'float64', 'O-Contact% (sc)': 'float64', 'Z-Contact% (sc)': 'float64',
                                 'Contact% (sc)': 'float64', 'Zone% (sc)': 'float64', 'Pace': 'float64',
                                 'RA9-WAR': 'float64', 'BIP-Wins': 'float64', 'LOB-Wins': 'float64',
                                 'FDP-Wins': 'float64', 'Age Rng': 'string', 'K-BB%': 'float64', 'Pull%': 'float64',
                                 'Cent%': 'float64', 'Oppo%': 'float64', 'Soft%': 'float64', 'Med%': 'float64',
                                 'Hard%': 'float64', 'kwERA': 'float64', 'TTO%': 'float64', 'CH% (pi)': 'float64',
                                 'CS% (pi)': 'float64', 'CU% (pi)': 'float64', 'FA% (pi)': 'float64',
                                 'FC% (pi)': 'float64', 'FS% (pi)': 'float64', 'KN% (pi)': 'float64',
                                 'SB% (pi)': 'float64', 'SI% (pi)': 'float64', 'SL% (pi)': 'float64',
                                 'XX% (pi)': 'float64', 'vCH (pi)': 'float64', 'vCS (pi)': 'float64',
                                 'vCU (pi)': 'float64', 'vFA (pi)': 'float64', 'vFC (pi)': 'float64',
                                 'vFS (pi)': 'float64', 'vKN (pi)': 'float64', 'vSB (pi)': 'float64',
                                 'vSI (pi)': 'float64', 'vSL (pi)': 'float64', 'vXX (pi)': 'float64',
                                 'CH-X (pi)': 'float64', 'CS-X (pi)': 'float64', 'CU-X (pi)': 'float64',
                                 'FA-X (pi)': 'float64', 'FC-X (pi)': 'float64', 'FS-X (pi)': 'float64',
                                 'KN-X (pi)': 'float64', 'SB-X (pi)': 'float64', 'SI-X (pi)': 'float64',
                                 'SL-X (pi)': 'float64', 'XX-X (pi)': 'float64', 'CH-Z (pi)': 'float64',
                                 'CS-Z (pi)': 'float64', 'CU-Z (pi)': 'float64', 'FA-Z (pi)': 'float64',
                                 'FC-Z (pi)': 'float64', 'FS-Z (pi)': 'float64', 'KN-Z (pi)': 'float64',
                                 'SB-Z (pi)': 'float64', 'SI-Z (pi)': 'float64', 'SL-Z (pi)': 'float64',
                                 'XX-Z (pi)': 'float64', 'wCH (pi)': 'float64', 'wCS (pi)': 'float64',
                                 'wCU (pi)': 'float64', 'wFA (pi)': 'float64', 'wFC (pi)': 'float64',
                                 'wFS (pi)': 'float64', 'wKN (pi)': 'float64', 'wSB (pi)': 'float64',
                                 'wSI (pi)': 'float64', 'wSL (pi)': 'float64', 'wXX (pi)': 'float64',
                                 'wCH/C (pi)': 'float64', 'wCS/C (pi)': 'float64', 'wCU/C (pi)': 'float64',
                                 'wFA/C (pi)': 'float64', 'wFC/C (pi)': 'float64', 'wFS/C (pi)': 'float64',
                                 'wKN/C (pi)': 'float64', 'wSB/C (pi)': 'float64', 'wSI/C (pi)': 'float64',
                                 'wSL/C (pi)': 'float64', 'wXX/C (pi)': 'float64', 'O-Swing% (pi)': 'float64',
                                 'Z-Swing% (pi)': 'float64', 'Swing% (pi)': 'float64', 'O-Contact% (pi)': 'float64',
                                 'Z-Contact% (pi)': 'float64', 'Contact% (pi)': 'float64', 'Zone% (pi)': 'float64',
                                 'Pace (pi)': 'float64', 'FRM': 'float64', 'K/9+': 'Int64', 'BB/9+': 'Int64',
                                 'K/BB+': 'Int64', 'H/9+': 'Int64', 'HR/9+': 'Int64', 'AVG+': 'Int64', 'WHIP+': 'Int64',
                                 'BABIP+': 'Int64', 'LOB%+': 'Int64', 'K%+': 'Int64', 'BB%+': 'Int64',
                                 'LD%+': 'float64', 'GB%+': 'float64', 'FB%+': 'float64', 'HR/FB%+': 'float64',
                                 'Pull%+': 'float64', 'Cent%+': 'float64', 'Oppo%+': 'float64', 'Soft%+': 'float64',
                                 'Med%+': 'float64', 'Hard%+': 'float64', 'EV': 'float64', 'LA': 'float64',
                                 'Barrels': 'float64', 'Barrel%': 'float64', 'maxEV': 'float64', 'HardHit': 'float64',
                                 'HardHit%': 'float64', 'Events': 'Int64', 'CStr%': 'float64', 'CSW%': 'float64',
                                 'xERA': 'float64'},
    'statcast_pitcher': {'pitch_type': 'string', 'game_date': 'datetime64', 'release_speed': 'float64',
                         'release_pos_x': 'float64', 'release_pos_z': 'float64', 'player_name': 'string',
                         'batter': 'Int64', 'pitcher': 'Int64', 'events': 'string', 'description': 'string',
                         'spin_dir': 'float64', 'spin_rate_deprecated': 'float64', 'break_angle_deprecated': 'float64',
                         'break_length_deprecated': 'float64', 'zone': 'Int64', 'des': 'string', 'game_type': 'string',
                         'stand': 'string', 'p_throws': 'string', 'home_team': 'string', 'away_team': 'string',
                         'type': 'string', 'hit_location': 'Int64', 'bb_type': 'string', 'balls': 'Int64',
                         'strikes': 'Int64', 'game_year': 'Int64', 'pfx_x': 'float64', 'pfx_z': 'float64',
                         'plate_x': 'float64', 'plate_z': 'float64', 'on_3b': 'Int64', 'on_2b': 'Int64',
                         'on_1b': 'Int64', 'outs_when_up': 'Int64', 'inning': 'Int64', 'inning_topbot': 'string',
                         'hc_x': 'float64', 'hc_y': 'float64', 'tfs_deprecated': 'float64',
                         'tfs_zulu_deprecated': 'float64', 'fielder_2': 'Int64', 'umpire': 'string', 'sv_id': 'string',
                         'vx0': 'float64', 'vy0': 'float64', 'vz0': 'float64', 'ax': 'float64', 'ay': 'float64',
                         'az': 'float64', 'sz_top': 'float64', 'sz_bot': 'float64', 'hit_distance_sc': 'Int64',
                         'launch_speed': 'float64', 'launch_angle': 'Int64', 'effective_speed': 'float64',
                         'release_spin_rate': 'Int64', 'release_extension': 'float64', 'game_pk': 'Int64',
                         'pitcher.1': 'Int64', 'fielder_2.1': 'Int64', 'fielder_3': 'Int64', 'fielder_4': 'Int64',
                         'fielder_5': 'Int64', 'fielder_6': 'Int64', 'fielder_7': 'Int64', 'fielder_8': 'Int64',
                         'fielder_9': 'Int64', 'release_pos_y': 'float64', 'estimated_ba_using_speedangle': 'float64',
                         'estimated_woba_using_speedangle': 'float64', 'woba_value': 'float64', 'woba_denom': 'float64',
                         'babip_value': 'float64', 'iso_value': 'float64', 'launch_speed_angle': 'float64',
                         'at_bat_number': 'Int64', 'pitch_number': 'Int64', 'pitch_name': 'string',
                         'home_score': 'Int64', 'away_score': 'Int64', 'bat_score': 'Int64', 'fld_score': 'Int64',
                         'post_away_score': 'Int64', 'post_home_score': 'Int64', 'post_bat_score': 'Int64',
                         'post_fld_score': 'Int64', 'if_fielding_alignment': 'string',
                         'of_fielding_alignment': 'string', 'spin_axis': 'float64', 'delta_home_win_exp': 'float64',
                         'delta_run_exp': 'float64'},
    'pitching': {'playerID': 'string', 'yearID': 'Int64', 'stint': 'Int64', 'teamID': 'string', 'lgID': 'string',
                 'W': 'Int64', 'L': 'Int64', 'G': 'Int64', 'GS': 'Int64', 'CG': 'Int64', 'SHO': 'Int64', 'SV': 'Int64',
                 'IPouts': 'Int64', 'H': 'Int64', 'ER': 'Int64', 'HR': 'Int64', 'BB': 'Int64', 'SO': 'Int64',
                 'BAOpp': 'float64', 'ERA': 'float64', 'IBB': 'float64', 'WP': 'Int64', 'HBP': 'float64', 'BK': 'Int64',
                 'BFP': 'float64', 'GF': 'Int64', 'R': 'Int64', 'SH': 'float64', 'SF': 'float64', 'GIDP': 'float64'},
    'team_pitching_bref': {'Pos': 'string', 'Name': 'string', 'Year': 'Int64', 'Age': 'Int64', 'W': 'Int64',
                           'L': 'Int64', 'W-L%': 'float64', 'ERA': 'float64', 'G': 'float64', 'GS': 'float64',
                           'GF': 'float64', 'CG': 'float64', 'SHO': 'float64', 'SV': 'float64', 'IP': 'float64',
                           'H': 'float64', 'R': 'float64', 'ER': 'float64', 'HR': 'float64', 'BB': 'float64',
                           'IBB': 'float64', 'SO': 'float64', 'HBP': 'float64', 'BK': 'float64', 'WP': 'float64',
                           'BF': 'float64', 'ERA+': 'float64', 'FIP': 'float64', 'WHIP': 'float64', 'H9': 'float64',
                           'HR9': 'float64', 'BB9': 'float64', 'SO9': 'float64', 'SO/W': 'float64'},
    'pitching_stats_range': {'Name': 'string', 'Age': 'Int64', '#days': 'Int64', 'Lev': 'string', 'Tm': 'string',
                             'G': 'Int64', 'GS': 'Int64', 'W': 'float64', 'L': 'float64', 'SV': 'float64',
                             'IP': 'float64', 'H': 'Int64', 'R': 'Int64', 'ER': 'Int64', 'BB': 'Int64', 'SO': 'Int64',
                             'HR': 'Int64', 'HBP': 'Int64', 'ERA': 'float64', 'AB': 'float64', '2B': 'float64',
                             '3B': 'float64', 'IBB': 'float64', 'GDP': 'float64', 'SF': 'float64', 'SB': 'float64',
                             'CS': 'float64', 'PO': 'float64', 'BF': 'float64', 'Pit': 'float64', 'Str': 'float64',
                             'StL': 'float64', 'StS': 'float64', 'GB/FB': 'float64', 'LD': 'float64', 'PU': 'float64',
                             'WHIP': 'float64', 'BAbip': 'float64', 'SO9': 'float64', 'SO/W': 'float64'},
    'statcast_pitcher_expected_stats': {'last_name': 'string', ' first_name': 'string', 'player_id': 'Int64',
                                        'pa': 'Int64', 'bip': 'Int64', 'ba': 'float64', 'est_ba': 'float64',
                                        'est_ba_minus_ba_diff': 'float64', 'slg': 'float64', 'est_slg': 'float64',
                                        'est_slg_minus_slg_diff': 'float64', 'woba': 'float64', 'est_woba': 'float64',
                                        'est_woba_minus_woba_diff': 'float64', 'era': 'float64', 'xera': 'float64',
                                        'era_minus_xera_diff': 'float64'},
    'pitching_post': {'playerID': 'string', 'yearID': 'Int64', 'round': 'string', 'teamID': 'string', 'lgID': 'string',
                      'W': 'Int64', 'L': 'Int64', 'G': 'Int64', 'GS': 'Int64', 'CG': 'Int64', 'SHO': 'Int64',
                      'SV': 'Int64', 'IPouts': 'Int64', 'H': 'Int64', 'ER': 'Int64', 'HR': 'Int64', 'BB': 'Int64',
                      'SO': 'Int64', 'BAOpp': 'float64', 'ERA': 'float64', 'IBB': 'float64', 'WP': 'float64',
                      'HBP': 'float64', 'BK': 'float64', 'BFP': 'float64', 'GF': 'Int64', 'R': 'Int64', 'SH': 'float64',
                      'SF': 'float64', 'GIDP': 'float64'},
    'bwar_pitch': {'name_common': 'string', 'age': 'Int64', 'mlb_ID': 'Int64', 'player_ID': 'string',
                   'year_ID': 'Int64', 'team_ID': 'string', 'stint_ID': 'Int64', 'lg_ID': 'string', 'G': 'Int64',
                   'GS': 'Int64', 'IPouts': 'Int64', 'IPouts_start': 'float64', 'IPouts_relief': 'float64',
                   'RA': 'Int64', 'xRA': 'float64', 'xRA_sprp_adj': 'float64', 'xRA_extras_adj': 'float64',
                   'xRA_def_pitcher': 'float64', 'PPF': 'Int64', 'PPF_custom': 'float64', 'xRA_final': 'float64',
                   'BIP': 'float64', 'BIP_perc': 'float64', 'RS_def_total': 'float64', 'runs_above_avg': 'float64',
                   'runs_above_avg_adj': 'float64', 'runs_above_rep': 'float64', 'RpO_replacement': 'float64',
                   'GR_leverage_index_avg': 'float64', 'WAR': 'float64', 'salary': 'float64', 'teamRpG': 'float64',
                   'oppRpG': 'float64', 'pyth_exponent': 'float64', 'waa_win_perc': 'float64', 'WAA': 'float64',
                   'WAA_adj': 'float64', 'oppRpG_rep': 'float64', 'pyth_exponent_rep': 'float64',
                   'waa_win_perc_rep': 'float64', 'WAR_rep': 'float64', 'ERA_plus': 'float64', 'ER_lg': 'float64'},
    'statcast': {'pitch_type': 'string', 'game_date': 'datetime64', 'release_speed': 'float64',
                 'release_pos_x': 'float64', 'release_pos_z': 'float64', 'player_name': 'string', 'batter': 'Int64',
                 'pitcher': 'Int64', 'events': 'string', 'description': 'string', 'spin_dir': 'float64',
                 'spin_rate_deprecated': 'float64', 'break_angle_deprecated': 'float64',
                 'break_length_deprecated': 'float64', 'zone': 'Int64', 'des': 'string', 'game_type': 'string',
                 'stand': 'string', 'p_throws': 'string', 'home_team': 'string', 'away_team': 'string',
                 'type': 'string', 'hit_location': 'Int64', 'bb_type': 'string', 'balls': 'Int64', 'strikes': 'Int64',
                 'game_year': 'Int64', 'pfx_x': 'float64', 'pfx_z': 'float64', 'plate_x': 'float64',
                 'plate_z': 'float64', 'on_3b': 'Int64', 'on_2b': 'Int64', 'on_1b': 'Int64', 'outs_when_up': 'Int64',
                 'inning': 'Int64', 'inning_topbot': 'string', 'hc_x': 'float64', 'hc_y': 'float64',
                 'tfs_deprecated': 'float64', 'tfs_zulu_deprecated': 'float64', 'fielder_2': 'Int64',
                 'umpire': 'string', 'sv_id': 'string', 'vx0': 'float64', 'vy0': 'float64', 'vz0': 'float64',
                 'ax': 'float64', 'ay': 'float64', 'az': 'float64', 'sz_top': 'float64', 'sz_bot': 'float64',
                 'hit_distance_sc': 'float64', 'launch_speed': 'float64', 'launch_angle': 'float64',
                 'effective_speed': 'float64', 'release_spin_rate': 'float64', 'release_extension': 'float64',
                 'game_pk': 'Int64', 'pitcher.1': 'Int64', 'fielder_2.1': 'Int64', 'fielder_3': 'Int64',
                 'fielder_4': 'Int64', 'fielder_5': 'Int64', 'fielder_6': 'Int64', 'fielder_7': 'Int64',
                 'fielder_8': 'Int64', 'fielder_9': 'Int64', 'release_pos_y': 'float64',
                 'estimated_ba_using_speedangle': 'float64', 'estimated_woba_using_speedangle': 'float64',
                 'woba_value': 'float64', 'woba_denom': 'float64', 'babip_value': 'float64', 'iso_value': 'float64',
                 'launch_speed_angle': 'float64', 'at_bat_number': 'Int64', 'pitch_number': 'Int64',
                 'pitch_name': 'string', 'home_score': 'Int64', 'away_score': 'Int64', 'bat_score': 'Int64',
                 'fld_score': 'Int64', 'post_away_score': 'Int64', 'post_home_score': 'Int64',
                 'post_bat_score': 'Int64', 'post_fld_score': 'Int64', 'if_fielding_alignment': 'string',
                 'of_fielding_alignment': 'string', 'spin_axis': 'string', 'delta_home_win_exp': 'float64',
                 'delta_run_exp': 'float64'},
    'statcast_pitcher_active_spin': {'last_name': 'string', ' first_name': 'string', 'pitch_hand': 'string',
                                     'active_spin_fourseam': 'float64', 'active_spin_sinker': 'float64',
                                     'active_spin_cutter': 'float64', 'active_spin_changeup': 'float64',
                                     'active_spin_fastball': 'float64', 'active_spin_slider': 'float64',
                                     'active_spin_curve': 'float64'},
    'statcast_pitcher_percentile_ranks': {'player_name': 'string', 'player_id': 'Int64', 'xwoba': 'float64',
                                          'xba': 'float64', 'xslg': 'float64', 'xiso': 'float64', 'xobp': 'float64',
                                          'brl': 'float64', 'brl_percent': 'float64', 'exit_velocity': 'float64',
                                          'hard_hit_percent': 'float64', 'k_percent': 'float64',
                                          'bb_percent': 'float64', 'whiff_percent': 'float64', 'xera': 'float64',
                                          'fb_velocity': 'float64', 'fb_spin': 'float64', 'curve_spin': 'float64'},
    'pitching_stats_bref': {'Name': 'string', 'Age': 'Int64', '#days': 'Int64', 'Lev': 'string', 'Tm': 'string',
                            'G': 'Int64', 'GS': 'Int64', 'W': 'float64', 'L': 'float64', 'SV': 'float64',
                            'IP': 'float64', 'H': 'Int64', 'R': 'Int64', 'ER': 'Int64', 'BB': 'Int64', 'SO': 'Int64',
                            'HR': 'Int64', 'HBP': 'Int64', 'ERA': 'float64', 'AB': 'Int64', '2B': 'Int64',
                            '3B': 'Int64', 'IBB': 'Int64', 'GDP': 'Int64', 'SF': 'Int64', 'SB': 'Int64', 'CS': 'Int64',
                            'PO': 'Int64', 'BF': 'Int64', 'Pit': 'Int64', 'Str': 'float64', 'StL': 'float64',
                            'StS': 'float64', 'GB/FB': 'float64', 'LD': 'float64', 'PU': 'float64', 'WHIP': 'float64',
                            'BAbip': 'float64', 'SO9': 'float64', 'SO/W': 'float64'},
    'statcast_pitcher_arsenal_stats': {'last_name': 'string', ' first_name': 'string', 'player_id': 'Int64',
                                       'team_name_alt': 'string', 'pitch_type': 'string', 'pitch_name': 'string',
                                       'run_value_per_100': 'float64', 'run_value': 'Int64', 'pitches': 'Int64',
                                       'pitch_usage': 'float64', 'pa': 'Int64', 'ba': 'float64', 'slg': 'float64',
                                       'woba': 'float64', 'whiff_percent': 'float64', 'k_percent': 'float64',
                                       'put_away': 'float64', 'est_ba': 'float64', 'est_slg': 'float64',
                                       'est_woba': 'float64', 'hard_hit_percent': 'float64'},
    'statcast_pitcher_pitch_arsenal': {'last_name': 'string', ' first_name': 'string', 'pitcher': 'Int64',
                                       'ff_avg_speed': 'float64', 'si_avg_speed': 'float64', 'fc_avg_speed': 'float64',
                                       'sl_avg_speed': 'float64', 'ch_avg_speed': 'float64', 'cu_avg_speed': 'float64',
                                       'fs_avg_speed': 'float64', 'kn_avg_speed': 'float64'},
    'statcast_pitcher_exitvelo_barrels': {'last_name': 'string', ' first_name': 'string', 'player_id': 'Int64',
                                          'attempts': 'Int64', 'avg_hit_angle': 'float64',
                                          'anglesweetspotpercent': 'float64', 'max_hit_speed': 'float64',
                                          'avg_hit_speed': 'float64', 'fbld': 'float64', 'gb': 'float64',
                                          'max_distance': 'float64', 'avg_distance': 'float64',
                                          'avg_hr_distance': 'float64', 'ev95plus': 'float64', 'ev95percent': 'float64',
                                          'barrels': 'float64', 'brl_percent': 'float64', 'brl_pa': 'float64'},
    'statcast_pitcher_pitch_movement': {'last_name': 'string', ' first_name': 'string', 'pitcher_id': 'Int64',
                                        'team_name': 'string', 'team_name_abbrev': 'string', 'pitch_hand': 'string',
                                        'avg_speed': 'float64', 'pitches_thrown': 'Int64', 'total_pitches': 'Int64',
                                        'pitches_per_game': 'float64', 'pitch_per': 'float64', 'pitch_type': 'string',
                                        'pitch_type_name': 'string', 'pitcher_break_z': 'float64',
                                        'league_break_z': 'float64', 'diff_z': 'float64', 'rise': 'Int64',
                                        'pitcher_break_x': 'float64', 'league_break_x': 'float64', 'diff_x': 'float64',
                                        'tail': 'Int64', 'percent_rank_diff_z': 'float64',
                                        'percent_rank_diff_x': 'float64'}}


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
    for year in range(START_YEAR, END_YEAR + 1):
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
            #     else:
            #         print("debug 1")
            #         print(d["games"])
            # else:
            #     print("debug 2")
            #     print(d)
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


# These methods are silly and don't include a year column when they should
def statcast_pitcher_active_spin(year: int, minP: int = 250, _type: str = 'spin-based') -> pd.DataFrame:
    data = sp_active_spin(year=year, minP=minP, _type=_type)
    data["year"] = year
    return data


def statcast_pitcher_arsenal_stats(year: int, minPA: int = 25) -> pd.DataFrame:
    data = sp_arsenal_stats(year=year, minPA=minPA)
    data["year"] = year
    return data


def yl(start_year=START_YEAR, end_year=END_YEAR):
    return list(range(start_year, min(CURRENT_YEAR + 1, end_year + 1)))


class MultiIterator:

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

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.indices)

    def add_failure(self, f):
        self.failure.append(f)

    def add_success(self, s):
        self.success.append(s)


class WeekIterable(MultiIterator):

    def __init__(self, i=[], columns=[], start_year=START_YEAR, end_year=END_YEAR):
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
            if game_week(start, start + sunday_td):
                weeks.append(pd.to_datetime(start))
            start += week_td
        return weeks

    def __next__(self):
        x = list(super().__next__())
        x[-2] = x[-2].to_pydatetime().date()
        x[-1] = x[-1].to_pydatetime().date()
        return tuple(x)


class YearIterator(MultiIterator):

    def __init__(self, keys, start_year=START_YEAR, end_year=END_YEAR):
        super().__init__(i=[yl(start_year, end_year)], columns=["year"])
        self.keys = list(keys)

    def __next__(self):
        (year,) = super().__next__()
        return not today(year), dict([(key, year) for key in self.keys])


class StatcastIterator(WeekIterable):

    def __init__(self, teams=team_ids, start_year=START_YEAR, end_year=END_YEAR):
        super().__init__(i=[teams], columns=["team"], start_year=start_year, end_year=end_year)

    def __next__(self):
        team, start, end = super().__next__()
        return not today(year=end.year, month=end.month, day=end.day), {
            "start_dt": "{}-{}-{}".format(start.year, start.month, start.day),
            "end_dt": "{}-{}-{}".format(end.year, end.month, end.day),
            "team": team,
            "verbose": False,
            "parallel": True
        }


class StatcastPitcherIterator(WeekIterable):

    def __init__(self, start_year=START_YEAR, end_year=END_YEAR):
        p = bwar_pitch(return_all=True)[["mlb_ID", "year_ID", "IPouts"]].astype({"mlb_ID": pd.Int64Dtype(), "year_ID": pd.Int64Dtype()}).rename(columns={"year_ID": "year", "mlb_ID": "player_id"})
        self.pitchers = p[(p.year >= start_year) & (p.year <= end_year)]
        self.pitchers = self.pitchers[self.pitchers["IPouts"] > 0]
        self.pitchers = self.pitchers.drop_duplicates(subset=["year", "player_id"])
        ids = list(pd.unique(self.pitchers["player_id"]))
        super().__init__(i=[ids], columns=["player_id"], start_year=start_year, end_year=end_year)

    def clean_df(self):
        super().clean_df()
        og_columns = list(self.df.columns)
        self.df["year"] = self.df["start_date"].dt.year
        merged = pd.merge(self.pitchers, self.df, left_on=["player_id", "year"], right_on=["player_id", "year"])
        self.df = merged[og_columns]

    def __next__(self):
        player_id, start, end = super().__next__()
        return not today(year=end.year, month=end.month, day=end.day), {
            "start_dt": "{}-{}-{}".format(start.year, start.month, start.day),
            "end_dt": "{}-{}-{}".format(end.year, end.month, end.day),
            "player_id": player_id
        }


class PitchingStatsIterator(WeekIterable):

    def __init__(self, start_year=START_YEAR, end_year=END_YEAR):
        super().__init__(start_year=start_year, end_year=end_year)

    def __next__(self):
        start, end = super().__next__()
        return not today(year=end.year, month=end.month, day=end.day), {
            "start_dt": "{}-{}-{}".format(start.year, start.month, start.day),
            "end_dt": "{}-{}-{}".format(end.year, end.month, end.day),
        }


class TeamPitchingIterator(MultiIterator):

    def __init__(self, teams=team_ids, start_year=START_YEAR, end_year=END_YEAR):
        super().__init__(i=[teams, yl(start_year, end_year)],  columns=["team", "year"])

    def clean_df(self):
        self.df = self.df.drop(self.df[(self.df["team"] == "TBD") & (self.df["year"] > 2007)].index)
        self.df = self.df.drop(self.df[(self.df["team"] == "TBA") & (self.df["year"] < 2008)].index)
        self.df = self.df.drop(self.df[(self.df["team"] == "ANA") & (self.df["year"] > 2004)].index)
        self.df = self.df.drop(self.df[(self.df["team"] == "WSN") & (self.df["year"] < 2005)].index)
        self.df = self.df.drop(self.df[(self.df["team"] == "FLA") & (self.df["year"] > 2012)].index)


    def __next__(self):
        team, year = super().__next__()
        return not today(year), {
            "team": team,
            "start_season": year,
            "end_season": year
        }


class StatcastPitcherPitchMovementIterator(MultiIterator):

    def __init__(self, pitches=pitch_types, start_year=START_YEAR, end_year=END_YEAR):
        super().__init__(i=[pitches, yl(start_year, end_year)], columns=["pitch", "year"])

    def __next__(self):
        pitch, year = super().__next__()

        return not today(year), {
            "pitch_type": pitch,
            "year": year,
        }


# A list of API methods and the iterable dicts of parameters used to "span" the API.
api_methods = {
    # playerid_reverse_lookup: [(True, {
    #     "player_ids": [477132],
    #     "key_type": "mlbam"
    # })],
    # player_search_list: [(True, {
    #     "player_list": [("kershaw", "clayton")]
    # })],
    # playerid_lookup: [(True, {
    #     "last": "kershaw",
    #     "first": "clayton",
    #     "fuzzy": False
    # })],
    # chadwick_register: [(False, {
    #     "save": True
    # })],
    # fangraphs_teams: [(False, {
    #     "season": None,
    #     "league": "ALL"
    # })],
    statcast: StatcastIterator(),
    # statcast_pitcher: StatcastPitcherIterator(),
    # pitching_stats_bref: YearIterator(keys=["season"], start_year=2008),
    # pitching_stats_range: PitchingStatsIterator(start_year=2008),
    # bwar_pitch: [(False, {
    #     "return_all": True
    # })],
    # pitching_stats: YearIterator(keys=["start_season", "end_season"]),
    # team_pitching_bref: TeamPitchingIterator(),
    # pitching: [{}],
    # pitching_post: [{}],
    # statcast_pitcher_exitvelo_barrels: YearIterator(keys=["year"]),
    # statcast_pitcher_expected_stats: YearIterator(keys=["year"]),
    # statcast_pitcher_pitch_arsenal: YearIterator(keys=["year"]),
    # statcast_pitcher_arsenal_stats: YearIterator(keys=["year"]),
    # statcast_pitcher_pitch_movement: StatcastPitcherPitchMovementIterator(),
    # statcast_pitcher_active_spin: YearIterator(keys=["year"]),
    # statcast_pitcher_percentile_ranks: YearIterator(keys=["year"]),
    # statcast_pitcher_spin_dir_comp: YearIterator(keys=["year"])
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
    # playerid_reverse_lookup.__qualname__: chadwick_cleanup,
    # player_search_list.__qualname__: chadwick_cleanup,
    # playerid_lookup.__qualname__: chadwick_cleanup,
    # chadwick_register.__qualname__: chadwick_cleanup,
    # fangraphs_teams.__qualname__: copy_year("yearID"),
    statcast.__qualname__: copy_year("game_year"),
    # statcast_pitcher.__qualname__: copy_year("game_year"),
    # bwar_pitch.__qualname__: copy_year("year_ID"),
    # pitching_stats.__qualname__: copy_year("Season"),
    # team_pitching_bref.__qualname__: copy_year("Year"),
    # pitching.__qualname__: copy_year("yearID"),
    # pitching_post.__qualname__: copy_year("yearID"),
    # statcast_pitcher_expected_stats.__qualname__: copy_year("year"),
    # statcast_pitcher_pitch_movement.__qualname__: copy_year("year"),
    # statcast_pitcher_percentile_ranks.__qualname__: copy_year("year"),
    # statcast_pitcher_spin_dir_comp.__qualname__: copy_year("year")
}

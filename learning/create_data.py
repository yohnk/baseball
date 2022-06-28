from os.path import join, exists
import pickle
from logging_config import log
import pandas as pd
import numpy as np
from pybaseball import playerid_lookup, chadwick_register

START_YEAR = 2016
#https://statsapi.mlb.com/api/v1/pitchTypes


def load_dataframes():
    log.info("Loading dataframes")
    with open(join("data", "build", "statcast_pitcher_percentile_ranks.pkl"), "rb") as f:
        percentile_ranks = pickle.load(f)
        percentile_ranks = percentile_ranks[percentile_ranks.year >= START_YEAR]
    with open(join("data", "build", "statcast_pitcher_active_spin.pkl"), "rb") as f:
        active_spin = pickle.load(f)
        active_spin = active_spin[active_spin.year >= START_YEAR]
    with open(join("data", "build", "statcast_pitcher_pitch_movement.pkl"), "rb") as f:
        movement = pickle.load(f)
        movement = movement[movement.year >= START_YEAR]
    with open(join("data", "build", "FangraphsDataTable.fetch.pkl"), "rb") as f:
        fangraph = pickle.load(f)
        fangraph = fangraph[fangraph.year >= START_YEAR]
    with open(join("data", "build", "statcast_pitcher.pkl"), "rb") as f:
        pitcher = pickle.load(f)
        pitcher = pitcher[pitcher.year >= START_YEAR]
        pitcher = pitcher.dropna(subset=["pitch_type"])
    log.info("Dataframes loaded")
    return percentile_ranks, active_spin, movement, pitcher, fangraph


def clean_pitches(pitcher):
    # The statcast "pitcher" list has many more pitches than the movement/active_spin list.
    # Here we're either dropping or mapping pitches, so we have the same list as movement/spin.
    log.info("Consolidating pitches")
    to_drop = ["IN", "EP", "PO", "SC", "FA", "CS", "KN"]
    pitcher = pitcher[~pitcher.pitch_type.isin(to_drop)]
    pitcher.loc[pitcher.pitch_type == "FO", "pitch_type"] = "FS"
    pitcher.loc[pitcher.pitch_type == "SI", "pitch_type"] = "SIFT"
    pitcher.loc[pitcher.pitch_type == "FT", "pitch_type"] = "SIFT"
    pitcher.loc[pitcher.pitch_type == "CU", "pitch_type"] = "CUKC"
    pitcher.loc[pitcher.pitch_type == "KC", "pitch_type"] = "CUKC"
    return pitcher


def lefty(pitcher):
    # The spin axis for left-handed pitchers is mirrored across the 12-6 (360-180) axis.
    log.info("Flipping L to R")
    pitcher.loc[pitcher.p_throws == "L", "spin_axis"] = (pitcher.loc[pitcher.p_throws == "L", "spin_axis"] - 360).abs()
    pitcher.loc[pitcher.p_throws == "L", "pfx_x"] = pitcher.loc[pitcher.p_throws == "L", "pfx_x"] * -1
    return pitcher


def rename_spin(active_spin):
    # Rename the active_spin columns to match our convention
    log.info("Renaming active_spin columns")
    return active_spin.rename(columns={
        "active_spin_fourseam": "FF_active_spin",
        "active_spin_slider": "SL_active_spin",
        "active_spin_curve": "CUKC_active_spin",
        "active_spin_changeup": "CH_active_spin",
        "active_spin_sinker": "SIFT_active_spin",
        "active_spin_cutter": "FC_active_spin",
        "active_spin_fastball": "FS_active_spin"
    })


def map_player_ids(movement, active_spin):

    # This method can take a while and results in the exact same thing. So lets cache if we can...
    cache_movement_path = join("learning", "tmp", "map_player_ids_movement.pkl")
    cache_spin_path = join("learning", "tmp", "map_player_ids_active_spin.pkl")
    if exists(cache_movement_path) and exists(cache_spin_path):
        with open(cache_movement_path, "rb") as cm, open(cache_spin_path, "rb") as cs:
            return pickle.load(cm), pickle.load(cs)

    # Make sure each table has a "player_id" column
    log.info("Finding player_ids for active_spin")
    movement = movement.rename(columns={"pitcher_id": "player_id"})
    active_spin["player_id"] = -1
    for idx, row in active_spin.iterrows():
        if row["player_id"] == -1:
            lookup = playerid_lookup(last=row["last_name"].strip(), first=row[" first_name"].strip(), fuzzy=False)
            if len(lookup) == 0:
                lookup = playerid_lookup(last=row["last_name"].strip(), first=row[" first_name"].strip(), fuzzy=True)
            lookup = lookup[lookup.key_mlbam != -1]
            if len(lookup) > 0:
                active_spin.loc[((active_spin['last_name'] == row["last_name"]) & (
                            active_spin[' first_name'] == row[" first_name"])), "player_id"] = lookup.iloc[0][
                    "key_mlbam"]
            else:
                log.error("Couldn't find MLB ID for {} {}".format(row[" first_name"], row["last_name"]))
        else:
            print("Skipping")

    unassigned = active_spin[active_spin.player_id == -1]
    log.info("active_spin rows without a player_id: {}".format(len(unassigned)))
    print("active_spin rows without a player_id")
    print(unassigned)
    active_spin = active_spin[active_spin.player_id != -1]

    with open(cache_movement_path, "wb") as cm, open(cache_spin_path, "wb") as cs:
        pickle.dump(movement, cm), pickle.dump(active_spin, cs)

    return movement, active_spin


def map_fangraph_id(fangraph):
    log.info("Merging fangraphs data")
    chad = chadwick_register()
    chad_columns = list(chad.columns)
    chad_columns.remove("key_mlbam")
    fangraph = pd.merge(left=chad, right=fangraph, left_on="key_fangraphs", right_on="IDfg")
    fangraph = fangraph.drop(chad_columns, axis=1)
    return fangraph.rename(columns={"key_mlbam": "player_id"})


def agg_pitchers(pitcher, pitches):

    # Roll up the pitch-by-pitch data in the pitcher table into aggregated statistics
    log.info("Aggregating stats for pitcher table")
    to_keep = ["release_spin_rate", "effective_speed", "spin_axis", "release_speed", "pfx_x", "pfx_z"]
    pitcher = pitcher.dropna(subset=to_keep)

    # Map zones to "strike_high", "strike_middle", "strike_low", "ball_high", "ball_low"
    merged_zones = ["strike_high", "strike_middle", "strike_low", "ball_high", "ball_low"]
    pitcher = pitcher.dropna(subset=["zone"])
    pitcher["merged_zone"] = pd.NA

    pitcher.loc[pitcher.zone == 11, "merged_zone"] = "ball_high"
    pitcher.loc[pitcher.zone == 12, "merged_zone"] = "ball_high"

    pitcher.loc[pitcher.zone == 1, "merged_zone"] = "strike_high"
    pitcher.loc[pitcher.zone == 2, "merged_zone"] = "strike_high"
    pitcher.loc[pitcher.zone == 3, "merged_zone"] = "strike_high"

    pitcher.loc[pitcher.zone == 4, "merged_zone"] = "strike_middle"
    pitcher.loc[pitcher.zone == 5, "merged_zone"] = "strike_middle"
    pitcher.loc[pitcher.zone == 6, "merged_zone"] = "strike_middle"

    pitcher.loc[pitcher.zone == 7, "merged_zone"] = "strike_low"
    pitcher.loc[pitcher.zone == 8, "merged_zone"] = "strike_low"
    pitcher.loc[pitcher.zone == 9, "merged_zone"] = "strike_low"

    pitcher.loc[pitcher.zone == 13, "merged_zone"] = "ball_low"
    pitcher.loc[pitcher.zone == 14, "merged_zone"] = "ball_low"

    columns = ["player_id", "year"]
    for pitch in pitches:
        for column in to_keep:
            columns.append(pitch + "_" + column + "_mean")
            columns.append(pitch + "_" + column + "_std")
        for zone in merged_zones:
            columns.append(pitch + "_" + zone)

    rows = []
    for player_id in pd.unique(pitcher.pitcher):
        player = pitcher[pitcher.pitcher == player_id]
        for year in pd.unique(player.year):
            player_year = player[player.year == year]
            row = [player_id, year]
            for pitch in pitches:
                player_pitch = player_year[player_year.pitch_type == pitch]
                if len(player_pitch) > 0:
                    for column in to_keep:
                        row.append(np.mean(player_pitch[column]))
                        row.append(np.std(player_pitch[column]))
                    zone_count = player_pitch['merged_zone'].value_counts()
                    for merged_zone in merged_zones:
                        if merged_zone in zone_count:
                            row.append(zone_count[merged_zone] / len(player_pitch))
                        else:
                            row.append(0.0)
                else:
                    row.extend([np.nan] * (len(to_keep) * 2))
                    row.extend([np.nan] * len(merged_zones))
            rows.append(row)

    log.info("Creating aggregated dataframe")
    return pd.DataFrame(rows, columns=columns).replace(pd.NA, np.nan)


def agg_movement(movement, pitches):
    # Roll up the pitch-by-pitch data in the pitcher table into aggregated statistics
    log.info("Aggregating stats for movement table")
    to_keep = ["pitcher_break_z", "pitcher_break_x", "rise", "tail"]
    columns = ["player_id", "year"]
    for pitch in pitches:
        for column in to_keep:
            columns.append(pitch + "_" + column)

    rows = []
    for player_id in pd.unique(movement.player_id):
        player = movement[movement.player_id == player_id]
        for year in pd.unique(player.year):
            player_year = player[player.year == year]
            row = [player_id, year]
            for pitch in pitches:
                player_pitch = player_year[player_year.pitch_type == pitch]
                if len(player_pitch) > 0:
                    for column in to_keep:
                        row.append(player_pitch.iloc[0][column])
                else:
                    for _ in to_keep:
                        row.append(np.nan)



            rows.append(row)

    log.info("Creating aggregated dataframe")
    return pd.DataFrame(rows, columns=columns).replace(pd.NA, np.nan)


def cache(percentile_ranks, active_spin, movement, pitcher):
    # Save tmp version
    log.info("Saving tmp versions")
    for file, df in [("pitcher.pkl", pitcher), ("movement_agg.pkl", movement), ("active_spin.pkl", active_spin),
                     ("percentile_ranks.pkl", percentile_ranks)]:
        with open(join("learning", "tmp", file), "wb") as f:
            pickle.dump(df, f)


def merge(percentile_ranks, active_spin, movement, pitcher, fangraph):
    log.info("Merging dataframes")
    merged = pd.merge(pitcher, active_spin, left_on=['player_id', 'year'], right_on=['player_id', 'year'])

    # The movement table seems to be missing data that we get from the pitchers table.
    # Keeping it around would require us to fill in that missing data and I don't think it's worthwhile.
    merged = pd.merge(merged, movement, left_on=['player_id', 'year'], right_on=['player_id', 'year'])

    merged = pd.merge(merged, percentile_ranks, left_on=['player_id', 'year'], right_on=['player_id', 'year'])
    return pd.merge(merged, fangraph, left_on=['player_id', 'year'], right_on=['player_id', 'year'])


def clean_all(merged, pitches):
    # We're using xwoba as a metric, so if it's nan lets drop it now
    required = ["xFIP", "FIP", "xwoba"]
    merged = merged.dropna(subset=required)

    # There are some rows where the "pitchers" table has values but the "active_spin" table doesn't.
    # We want to use the active_spin data, so drop cases where one is null and the other is not.
    # There's a 2nd legit case where the pitcher doesn't throw this pitch, so both will be null.
    for pitch in pitches:
        merged = merged[(~pd.isnull(merged[pitch + "_active_spin"]) & ~pd.isnull(merged[pitch + "_effective_speed_mean"])) | (pd.isnull(merged[pitch + "_active_spin"]) & pd.isnull(merged[pitch + "_effective_speed_mean"]))]

    # Above knocks out all the "FS" pitches
    to_drop = []
    for column in merged.columns:
        if column.startswith("FS_"):
            to_drop.append(column)

    merged = merged.drop(column, axis=1)

    return merged


def main():
    percentile_ranks, active_spin, movement, pitcher, fangraph = load_dataframes()
    pitcher = clean_pitches(pitcher)
    pitches = pd.unique(pitcher.pitch_type)
    pitcher = lefty(pitcher)
    active_spin = rename_spin(active_spin)
    fangraph = map_fangraph_id(fangraph)
    movement, active_spin = map_player_ids(movement, active_spin)
    pitcher = agg_pitchers(pitcher, pitches)
    movement = agg_movement(movement, pitches)
    merged = merge(percentile_ranks, active_spin, movement, pitcher, fangraph)

    # with open(join("learning", "combined.pkl"), "rb") as f:
    #     merged = pickle.load(f)

    merged = clean_all(merged, pitches)

    log.info("Saving dataframe")
    with open(join("learning", "combined.pkl"), "wb") as f:
        pickle.dump(merged, f)


if __name__ == "__main__":
    main()

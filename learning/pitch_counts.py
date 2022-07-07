import pickle
from os.path import join

import numpy as np
import pandas as pd


def clean_and_print(counter):
    to_remove = []
    for key in counter:
        if counter[key] == 0:
            to_remove.append(key)

    for key in to_remove:
        del counter[key]

    sorted_counter = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))

    for key in sorted_counter:
        print("{} -> {}".format(key, sorted_counter[key]))
    print()


def new_counter(pitches):
    counter = {}
    for pitch in pitches:
        counter[pitch] = 0
    return counter


with open(join("learning", "build", "combined.pkl"), "rb") as f:
    df = pickle.load(f)

pitches = ["FF", "SL", "CUKC", "CH", "SIFT", "FC", "FS"]

total = {}

for year in range(2016, 2022 + 1):
    year_counter = new_counter(pitches)
    year_df = df[df.year == year]
    total_pitches = np.sum(year_df.total)
    for player_id in pd.unique(df.player_id):
        player_year = year_df[year_df.player_id == player_id]
        if len(player_year) == 1:
            p = player_year.iloc[0]
            for pitch in pitches:
                if p[pitch + "_total"] > 0.0:
                    year_counter[pitch] += p[pitch + "_total"]
    for key in year_counter:
        year_counter[key] = year_counter[key] / total_pitches
    total[year] = year_counter
    print(year)
    clean_and_print(year_counter)

columns = ["year"]
columns.extend(pitches)
rows = []

for year in total:
    row = [year]
    for pitch in pitches:
        row.append(total[year][pitch])
    rows.append(row)

df = pd.DataFrame(rows, columns=columns)
df.to_csv(join("learning", "build", "pitch_counts.csv"))
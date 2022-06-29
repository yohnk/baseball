# Creating datatype list
# import csv
#
# types = {}
# type_map = {'int': "Int64", 'float': "float64", 'str': "string", 'bool': "boolean", 'date': "datetime64"}
#
#
# with open("data_types.csv", "r") as f:
#     reader = csv.reader(f)
#     # Read the header
#     reader.__next__()
#     for line in reader:
#         method, parameter, output, type = line[0], line[1], line[2], line[3]
#         if output == "TRUE":
#             if method not in types:
#                 types[method] = {}
#             method_dict = types[method]
#             method_dict[parameter] = type_map[type]
#
# print(types)

# import lzma
# from glob import glob
# from os.path import join
#
#
# def decomp(path):
#     with lzma.open(path, "rb") as orig, open(path.replace(".xz", ""), 'wb') as dest:
#         dest.write(orig.read())
#
#
# files = glob(join("data", "raw", "*.pkl.xz"))
# for file in files:
#     print(file)
#     decomp(file)

import pickle

import numpy as np
import pandas as pd
from numpy import count_nonzero
from pybaseball import chadwick_register

with open("data/build/statcast_pitcher.pkl", "rb") as f:
    pitcher = pickle.load(f)

lefty = pitcher[pitcher.p_throws == "L"]
righty = pitcher[pitcher.p_throws == "R"]

print("Lefty Mean", np.mean(lefty.loc[lefty.pitch_type == "FC", "plate_x"]))
print("Righty Mean", np.mean(righty.loc[righty.pitch_type == "FC", "plate_x"]))

# chad = chadwick_register()
#
# pd.merge(left=chad, right=df, left_on="key_fangraphs", right_on="IDfg")
#
# print(len(df))
#
# columns = list(df.columns)
#
# for column in sorted(columns):
#     series = df[column]
#     print(column, count_nonzero(series.isna()) / series.size)

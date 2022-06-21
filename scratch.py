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

with open("data/raw/statcast.pkl", "rb") as f:
    df = pickle.load(f)

series = df["release_speed"].loc[0:100]
print(series)

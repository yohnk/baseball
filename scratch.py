import lzma
import pickle
from glob import glob
from os.path import join

from pybaseball import fangraphs_teams, chadwick_register

file_names = glob(join("data", "build", "*.pkl.xz"))
for file_name in file_names:
    with lzma.open(file_name, "rb") as f:
        df = pickle.load(f)
    print(list(df.columns))

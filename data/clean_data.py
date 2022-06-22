import pickle
import traceback

from boilerplate import cleanup_methods, data_types
from glob import glob
from os.path import join, basename, exists
from logging_config import log
import pandas as pd
from numpy import nan

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)


def main():
    file_list = glob(join("data", "raw", "*.pkl"))
    for file in file_list:
        path = join("data", "build", basename(file))
        if not exists(path):
            log.info(file)
            with open(file, "rb") as f:
                df = pickle.load(f)

            method = basename(file).replace(".pkl", "")

            if method in cleanup_methods:
                log.info("Cleanup Methods")
                df = cleanup_methods[method](df)
            if method in data_types:
                log.info("Datatype Methods")
                data_type = data_types[method]
                df = df.replace(r'^\s*$', nan, regex=True)
                df = df.replace(pd.NA, nan)
                df = df.astype(data_type)

            log.info(df.dtypes)
            log.info("Saving...")
            with open(path, "wb") as f:
                pickle.dump(df, f)


if __name__ == "__main__":
    main()

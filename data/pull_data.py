import pandas as pd
import pickle

import pybaseball

from boilerplate import api_methods, cleanup_methods, dmd5
import lzma
import logging_config
import logging
from os.path import join, exists

import pybaseball
chadwick = pybaseball.chadwick_register()


def main():
    for method in api_methods.keys():
        all_dataframes = []
        for d in api_methods[method]:
            logging.info("Request: {} - {}".format(method.__qualname__, d))
            try:
                path = join("data", "cache", "{}_{}.pkl".format(method.__qualname__, dmd5(d)))
                if exists(path):
                    with open(path, "rb") as file:
                        df = pickle.load(file)
                else:
                    logging.info("Cache Miss")
                    df = method(**d)
                    with open(path, "wb") as file:
                        pickle.dump(df, file)
                    logging.info("Cached request to {}".format(path))
                all_dataframes.append(df.reset_index())
            except:
                logging.exception("Failed to get df")

        if len(all_dataframes) > 0:
            logging.info("Concatenating dataframes for {}".format(method.__qualname__))
            ret = pd.concat(all_dataframes)
            ret = ret.reset_index()
            if method in cleanup_methods:
                ret = cleanup_methods[method](ret)

            path = join("data", "build", method.__qualname__ + ".pkl.xz")
            with lzma.open(path, "w") as file:
                pickle.dump(ret, file)

            logging.info("Created {}".format(path))
        else:
            logging.info("No returns for {}".format(method.__qualname__))


if __name__ == "__main__":
    main()

import pandas as pd
import pickle
from boilerplate import api_methods, dmd5
from logging_config import log
from os.path import join, exists

import pybaseball
chadwick = pybaseball.chadwick_register()


def main():
    for method in api_methods.keys():
        path = join("data", "raw", method.__qualname__ + ".pkl")
        if not exists(path):
            all_dataframes = []
            for d in api_methods[method]:
                log.info("Request: {} - {}".format(method.__qualname__, d))
                try:
                    path = join("data", "cache", "{}_{}.pkl".format(method.__qualname__, dmd5(d)))
                    if exists(path):
                        with open(path, "rb") as file:
                            df = pickle.load(file)
                    else:
                        log.info("Cache Miss")
                        df = method(**d)
                        with open(path, "wb") as file:
                            pickle.dump(df, file)
                        log.info("Cached request to {}".format(path))
                    all_dataframes.append(df.reset_index())
                except:
                    log.exception("Failed to get df")

            if len(all_dataframes) > 0:
                log.info("Concatenating dataframes for {}".format(method.__qualname__))
                ret = pd.concat(all_dataframes)
                ret = ret.reset_index()
                with open(path, "wb") as file:
                    pickle.dump(ret, file)

                log.info("Created {}".format(path))
            else:
                log.info("No returns for {}".format(method.__qualname__))


if __name__ == "__main__":
    main()

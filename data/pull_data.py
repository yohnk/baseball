import pandas as pd
import pickle
from boilerplate import api_methods, dmd5
import lzma
import logging_config
import logging
from os.path import join, exists


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
                all_dataframes.append(df)
            except:
                logging.exception("Failed to get df")

        if len(all_dataframes) > 0:
            logging.info("Concatenating dataframes for {}".format(method.__qualname__))
            ret = pd.concat(all_dataframes)
            ret = ret.reset_index()

            path = join("data", "build", method.__qualname__ + ".pkl.xz")
            with lzma.open(path, "w") as file:
                pickle.dump(ret, file)

            logging.info("Created {}".format(path))
        else:
            logging.info("No returns for {}".format(method.__qualname__))


if __name__ == "__main__":
    main()

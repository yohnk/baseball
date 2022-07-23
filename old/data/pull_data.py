from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pickle
from boilerplate import api_methods, dmd5
from old.logging_config import log
from os.path import join, exists

import pybaseball
chadwick = pybaseball.chadwick_register()


def run_request(method, d, cache):
    log.debug("Request: {} - {}".format(method.__qualname__, d))
    try:
        path = join("data", "cache", "{}_{}.pkl".format(method.__qualname__, dmd5(d)))

        if cache and exists(path):
            with open(path, "rb") as file:
                df = pickle.load(file)
        else:
            df = method(**d)
            with open(path, "wb") as file:
                pickle.dump(df, file)
            log.debug("Cached request to {}".format(path))

        return df, d
    except:
        return None, d


def run_method(method, executor):
    fpath = join("data", "raw", method.__qualname__ + ".pkl")
    all_dataframes = []
    futures = []
    m_iter = api_methods[method]

    total = len(m_iter)
    count = 0

    for cache, d in m_iter:
        futures.append(executor.submit(run_request, method, d, cache))
    for futures in as_completed(futures):
        df, d = futures.result()
        if df is None:
            m_iter.add_failure(d)
        else:
            all_dataframes.append(df)
            m_iter.add_success(d)
        count += 1
        print("{}/{}".format(count, total))

    if len(all_dataframes) > 0:
        log.info("Concatenating dataframes for {}".format(method.__qualname__))
        ret = pd.concat(all_dataframes)
        ret = ret.reset_index()
        with open(fpath, "wb") as file:
            pickle.dump(ret, file)

        log.info("Created {}".format(fpath))
    else:
        log.info("No returns for {}".format(method.__qualname__))

    # if hasattr(m_iter, "success"):
    #     log.info("Success for {}".format(method.__qualname__))
    #     log.info(m_iter.success)

    if hasattr(m_iter, "failure"):
        log.error("Failures for {}".format(method.__qualname__))
        for f in m_iter.failure:
            log.error(f)


def main():
    executor = ProcessPoolExecutor(max_workers=3)

    for method in api_methods.keys():
        run_method(method, executor)

    log.info("Shutting down executor")
    executor.shutdown()


if __name__ == "__main__":
    main()

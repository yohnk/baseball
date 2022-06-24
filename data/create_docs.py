import inspect
from logging_config import log
from glob import glob
from os.path import join, basename
import pandas as pd
from pandas import unique
from pandas.api.types import is_numeric_dtype
from numpy import max, min, mean, median, count_nonzero, std
from boilerplate import api_methods
import pickle
import csv


def create_descriptions():
    out = {}
    with open(join("data", "descriptions.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)
        for l in reader:
            out[l[0]] = l[1]
    return out


def read_df(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def find_method(name):
    for method in api_methods.keys():
        if method.__qualname__ == name:
            return method
    return None


def nq(method, series):
    if is_numeric_dtype(series):
        try:
            return method(series)
        except TypeError:
            pass
    return None


def unique(series):
    u = pd.unique(series)
    if len(u) > 50:
        return None
    else:
        return str(list(u))


def year(df, col):
    if "year" not in df.columns:
        return None, None

    series = df[[col, "year"]]
    years = series.dropna(subset=[col])["year"]
    return years.min(), years.max()


def main():
    descriptions = create_descriptions()
    with open(join("data", "csv", "api.csv"), "w") as file:
        writer = csv.writer(file)
        ignore = ["level_0", "index", "year"]
        fields = ["method", "parameter", "output", "type", "max", "min", "mean", "median", "std", "%nan", "first_year", "last_year", "items", "description"]
        writer.writerow(fields)

        file_names = glob(join("data", "build", "*.pkl"))
        for file_name in file_names:
            rows = []
            log.info(file_name)
            method_name = basename(file_name).replace(".pkl", "")
            method = find_method(method_name)
            dataframe = read_df(file_name)

            # Add all the input rows
            argspec = inspect.getfullargspec(method)
            for arg in argspec.args:
                t = None if arg not in argspec.annotations else str(argspec.annotations[arg])
                rows.append([method_name, arg, False, t, None, None, None, None, None, None, None, None, None])

            # Add the output rows
            if dataframe.size > 0:
                for column in dataframe.columns:
                    if column not in ignore:
                        log.info(column)

                        desc = None
                        if column in descriptions:
                            desc = descriptions[column]

                        series = dataframe[column]
                        t = dataframe.dtypes.loc[[column]].values[0].name
                        yrange = year(dataframe, column)
                        rows.append([method.__qualname__, column, True, t, nq(max, series), nq(min, series),
                                     nq(mean, series), nq(median, series), nq(std, series), count_nonzero(series.isna()) / series.size,
                                     yrange[0], yrange[1], unique(series), desc])

            writer.writerows(rows)


if __name__ == "__main__":
    main()
    print("Done")

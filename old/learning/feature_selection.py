import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import join, exists
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from old.logging_config import log
from multiprocessing import cpu_count


def importance(x, y, classification=True, estimators=[], scoring=None,
               rfe_max_iter=1000, rfe_min_iter=10, rfe_thresh=0.1,
               est_max_iter=100, est_min_iter=3, est_thresh=0.01,
               n_jobs=-1, cv=3, metadata=None):
    if estimators is None or len(estimators) == 0:
        if classification:
            estimators = [DecisionTreeClassifier()]
        else:
            estimators = [DecisionTreeRegressor()]

    if scoring is None:
        if classification:
            scoring = "f1_weighted"
        else:
            # We want a scorer where 0 is the worst, inf is the best
            def regression_scorer(y_true, y_pred):
                mse = mean_squared_error(y_true, y_pred)
                try:
                    return 1 / mse
                except ZeroDivisionError:
                    return float("inf")

            scoring = make_scorer(regression_scorer)

    scores = np.zeros(x.shape[1], dtype=float)

    # Keep running the data through RFECV until we hit max iterations or the change is below a threshold
    log.info("RFECV")
    for i in range(1, rfe_max_iter + 1):
        tmp_scores = np.zeros(scores.shape, dtype=float)
        for e in estimators:
            selector = RFECV(estimator=e, scoring=scoring, n_jobs=n_jobs, cv=cv)
            selector.fit(x, y)
            best_score = max(selector.cv_results_["mean_test_score"])
            tmp_scores[selector.support_] += best_score
        tmp_scores = tmp_scores / len(estimators)

        # The new scores the old scores weighted a ((n - 1) / n) plus the new scores weighted at (1 / n)
        new = (scores * ((i - 1) / i)) + (tmp_scores / i)
        diff = np.sum(np.abs(scores - new))
        if i % 100 == 0:
            log.info("\t{} -> {} -> Diff: {}".format(metadata, i, diff))
        else:
            log.debug("\t{} -> {} -> Diff: {}".format(metadata, i, diff))
        scores = new
        if diff < rfe_thresh and i >= rfe_min_iter:
            break

    # Now sort the features based on score and one by one add them to our estimator
    fe_columns, fe_scores = zip(*sorted(zip(x.columns, scores), key=lambda x: x[1], reverse=True))
    running_columns = []
    col_scores = []
    log.debug("Estimator")
    for column in fe_columns:
        log.debug(column)
        running_columns.append(column)
        col_score = 0
        for i in range(1, est_max_iter + 1):
            tmp_scores = []
            for e in estimators:
                tmp_scores.append(np.mean(
                    cross_validate(estimator=e, X=x[running_columns], y=y, cv=cv, n_jobs=n_jobs, scoring=scoring)[
                        "test_score"]))
            new = (col_score * ((i - 1) / i)) + (np.mean(tmp_scores) / i)
            diff = np.abs(col_score - new)
            col_score = new
            if diff < est_thresh and i >= est_min_iter:
                break
        log.info("{} -> Score: {}".format(column, col_score))
        col_scores.append(col_score)

    return {
        "metadata": metadata,
        "fe_columns": fe_columns,
        "fe_scores": fe_scores,
        "est_scores": col_scores
    }


def main():
    with open(join("learning", "build", "combined.pkl"), "rb") as f:
        master_df = pickle.load(f)

    columns = ['FF_pct', 'FF_release_spin_rate_mean', 'FF_release_spin_rate_std', 'FF_effective_speed_mean', 'FF_effective_speed_std', 'FF_spin_axis_mean', 'FF_spin_axis_std', 'FF_release_speed_mean', 'FF_release_speed_std', 'FF_pfx_x_mean', 'FF_pfx_x_std', 'FF_pfx_z_mean', 'FF_pfx_z_std', 'FF_plate_x_mean', 'FF_plate_x_std', 'FF_plate_z_mean', 'FF_plate_z_std', 'SL_pct', 'SL_release_spin_rate_mean', 'SL_release_spin_rate_std', 'SL_effective_speed_mean', 'SL_effective_speed_std', 'SL_spin_axis_mean', 'SL_spin_axis_std', 'SL_release_speed_mean', 'SL_release_speed_std', 'SL_pfx_x_mean', 'SL_pfx_x_std', 'SL_pfx_z_mean', 'SL_pfx_z_std', 'SL_plate_x_mean', 'SL_plate_x_std', 'SL_plate_z_mean', 'SL_plate_z_std', 'CUKC_pct', 'CUKC_release_spin_rate_mean', 'CUKC_release_spin_rate_std', 'CUKC_effective_speed_mean', 'CUKC_effective_speed_std', 'CUKC_spin_axis_mean', 'CUKC_spin_axis_std', 'CUKC_release_speed_mean', 'CUKC_release_speed_std', 'CUKC_pfx_x_mean', 'CUKC_pfx_x_std', 'CUKC_pfx_z_mean', 'CUKC_pfx_z_std', 'CUKC_plate_x_mean', 'CUKC_plate_x_std', 'CUKC_plate_z_mean', 'CUKC_plate_z_std', 'CH_pct', 'CH_release_spin_rate_mean', 'CH_release_spin_rate_std', 'CH_effective_speed_mean', 'CH_effective_speed_std', 'CH_spin_axis_mean', 'CH_spin_axis_std', 'CH_release_speed_mean', 'CH_release_speed_std', 'CH_pfx_x_mean', 'CH_pfx_x_std', 'CH_pfx_z_mean', 'CH_pfx_z_std', 'CH_plate_x_mean', 'CH_plate_x_std', 'CH_plate_z_mean', 'CH_plate_z_std', 'SIFT_pct', 'SIFT_release_spin_rate_mean', 'SIFT_release_spin_rate_std', 'SIFT_effective_speed_mean', 'SIFT_effective_speed_std', 'SIFT_spin_axis_mean', 'SIFT_spin_axis_std', 'SIFT_release_speed_mean', 'SIFT_release_speed_std', 'SIFT_pfx_x_mean', 'SIFT_pfx_x_std', 'SIFT_pfx_z_mean', 'SIFT_pfx_z_std', 'SIFT_plate_x_mean', 'SIFT_plate_x_std', 'SIFT_plate_z_mean', 'SIFT_plate_z_std', 'FC_pct', 'FC_release_spin_rate_mean', 'FC_release_spin_rate_std', 'FC_effective_speed_mean', 'FC_effective_speed_std', 'FC_spin_axis_mean', 'FC_spin_axis_std', 'FC_release_speed_mean', 'FC_release_speed_std', 'FC_pfx_x_mean', 'FC_pfx_x_std', 'FC_pfx_z_mean', 'FC_pfx_z_std', 'FC_plate_x_mean', 'FC_plate_x_std', 'FC_plate_z_mean', 'FC_plate_z_std']

    executor = ProcessPoolExecutor(max_workers=cpu_count() - 1)
    futures = []

    for year in pd.unique(master_df["year"]):
        year_df = master_df[master_df.year == year]
        x = pd.DataFrame(PowerTransformer().fit_transform(year_df[columns].fillna(0)), columns=columns)
        y = year_df['FIP']
        futures.append(executor.submit(importance, x, y, classification=False, rfe_thresh=0.001, rfe_max_iter=100000, est_thresh=0.001, metadata=year))

    for f in as_completed(futures):
        results = f.result()

        year = results["metadata"]

        with open(join("learning", "build", "importance_results_{}.pkl".format(year)), "wb") as f:
            pickle.dump(results, f)

        plt.figure(figsize=(20, 5))
        plt.tight_layout()
        plt.plot(results["fe_columns"], results["fe_scores"])
        plt.plot(results["fe_columns"], results["est_scores"])
        ax = plt.gca()
        ax.set_xticklabels(labels=results["fe_columns"], rotation=90)
        plt.savefig(join("learning", "build", "info_{}.png".format(year)), bbox_inches='tight')

    executor.shutdown(wait=False)


def create_spreadsheet():
    with open(join("learning", "build", "combined.pkl"), "rb") as f:
        years = sorted(list(pd.unique(pickle.load(f)["year"])))

    with open(join("learning", "build", "importance_results_{}.pkl".format(years[1])), "rb") as f:
        features = list(pickle.load(f)["fe_columns"])

    results = {}
    for year in years:
        year_result = {}
        path = join("learning", "build", "importance_results_{}.pkl".format(year))
        if exists(path):
            with open(path, "rb") as f:
                tmp = pickle.load(f)
                for c in tmp["fe_columns"]:
                    year_result[c] = tmp["fe_scores"][tmp["fe_columns"].index(c)]
            results[year] = year_result

    rows = []
    for feature in features:
        row = [feature]
        for year in years:
            if year in results:
                row.append(results[year][feature])
        rows.append(row)

    columns = ["feature"]
    for year in years:
        if year in results:
            columns.append(year)

    df = pd.DataFrame(rows, columns=columns)

    for year in years:
        if year in results:
            df[year] = df[year] / df[year].sum()

    df.to_csv(join("learning", "build", "importance_year.csv"))


if __name__ == "__main__":
    # main()
    create_spreadsheet()

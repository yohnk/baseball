import pickle
from os.path import join

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from logging_config import log


def importance(x, y, classification=True, estimators=[], scoring=None,
               rfe_max_iter=1000, rfe_min_iter=10, rfe_thresh=0.1,
               est_max_iter=100, est_min_iter=3, est_thresh=0.01,
               n_jobs=-1, cv=3):
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
        log.info("\t{} -> Diff: {}".format(i, diff))
        scores = new
        if diff < rfe_thresh and i >= rfe_min_iter:
            break

    # Now sort the features based on score and one by one add them to our estimator
    fe_columns, fe_scores = zip(*sorted(zip(x.columns, scores), key=lambda x: x[1], reverse=True))
    running_columns = []
    col_scores = []
    log.info("Estimator")
    for column in fe_columns:
        log.info(column)
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
        "fe_columns": fe_columns,
        "fe_scores": fe_scores,
        "est_scores": col_scores
    }


def main():
    with open(join("learning", "build", "combined.pkl"), "rb") as f:
        master_df = pickle.load(f)
        print(list(master_df.columns))

    bins = {
        "docile": 2,
        "tercile": 3,
        "quartile": 4,
        "quintile": 5,
        "sextile": 6,
        "septile": 7,
        "octile": 8,
        "nonile": 9,
        "decile": 10
    }

    for tile in bins:
        master_df[tile] = pd.qcut(master_df['xFIP'], bins[tile], labels=False)

    suffixed_columns = ["release_spin_rate", "effective_speed", "spin_axis", "release_speed", "pfx_x", "pfx_z",
                        "plate_x", "plate_z"]
    # strike_zone = ["strike_high", "strike_middle", "strike_low", "ball_high", "ball_low"]
    suffixes = ["mean", "std"]
    pitches = ["FF", "SL", "CUKC", "CH", "SIFT", "FC"]

    columns = []
    for pitch in pitches:
        columns.append(pitch + "_pct")
        for sc in suffixed_columns:
            for suffix in suffixes:
                columns.append(pitch + "_" + sc + "_" + suffix)

    for c in columns:
        if c not in master_df.columns:
            print("Missing", c)

    x = pd.DataFrame(PowerTransformer().fit_transform(master_df[columns].fillna(0)), columns=columns)
    y = master_df['xFIP']

    results = importance(x, y, classification=False, rfe_thresh=0.1, est_thresh=0.1)
    with open(join("learning", "build", "importance_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    plt.figure(figsize=(20, 5))
    plt.tight_layout()
    plt.plot(results["fe_columns"], results["fe_scores"])
    plt.plot(results["fe_columns"], results["est_scores"])
    ax = plt.gca()
    ax.set_xticklabels(labels=results["fe_columns"], rotation=90)
    plt.savefig(join("learning", "build", "info.png"), bbox_inches='tight')


if __name__ == "__main__":
    main()

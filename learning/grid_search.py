import pickle
import warnings
from os.path import join

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.exceptions import DataConversionWarning
from learning.custom_learners import DTAdaBoost, SVCAdaBoost, MLPClassWrapper, VotingAdaBoost, MSDVotingClassifier
from logging_config import log


def frange(start=0, end=1, step=0.1):
    out = [start]
    while out[-1] <= end:
        out.append(out[-1] + step)
    return out


def irange(start=0, end=10, step=1):
    return list(range(start, end + 1, step))


def main():
    with open(join("learning", "combined.pkl"), "rb") as f:
        master_df = pickle.load(f)

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

    columns = ['FF_effective_speed_mean', 'CH_pfx_z_mean', 'FF_release_speed_std', 'FF_release_speed_mean', 'CUKC_pfx_z_mean', 'CH_effective_speed_std', 'FF_spin_axis_std', 'FF_pfx_x_mean', 'FF_spin_axis_mean', 'FF_pfx_z_mean', 'CH_spin_axis_mean', 'CH_effective_speed_mean']


    x = pd.DataFrame(PowerTransformer().fit_transform(master_df[columns].fillna(0)), columns=columns)
    y = master_df['tercile']

    log.info("Testing Voting Classifier")

    # DecisionTreeClassifier() Best Score 0.3315481986368062 {'ccp_alpha': 0.01, 'learning_rate': 0.9999999999999999, 'max_depth': 10, 'max_features': 20, 'n_estimators': 45}
    learners = {
        DTAdaBoost(): {
            "n_estimators": [50],
            "learning_rate": [0.3],
            "algorithm": ['SAMME.R'],
            "criterion": ["gini"],
            "splitter": ["best"],
            "max_depth": [20],
            "min_samples_split": [20],
            "min_samples_leaf": [3],
            "max_features": irange(1, 10, 1),
            "max_leaf_nodes": irange(100, 1000, 100),
            "ccp_alpha": [0.01]
        },
        # SVCAdaBoost(): {
        #     "n_estimators": [50],
        #     "learning_rate": [0.3],
        #     "algorithm": ['SAMME.R'],
        #     "C": [0.1],
        #     "kernel": ['linear'],
        #     "degree": [3],
        #     "gamma": ['scale'],
        #     "coef0": [0.0],
        #     "shrinking": [True],
        #     "probability": [True],
        #     "tol": [0.0008],
        #     "cache_size": [1000],
        #     "class_weight": [None],
        #     "max_iter": [-1],
        #     "decision_function_shape": ["ovr"],
        #     "break_ties": [False]
        # },
        # MLPClassWrapper(): {
        #     "hidden_layer_dimension": [13],
        #     "hidden_layer_value": [5],
        #     "activation": ["logistic"],
        #     "solver": ['adam'],
        #     "alpha": [0.001],
        #     "learning_rate": ['adaptive'],
        #     "learning_rate_init": [0.0025]
        # },
        VotingAdaBoost(): {
            "n_estimators": [50],
            "learning_rate": [0.9],
            "algorithm": ['SAMME.R'],
            "mlp_w": frange(0.0, 2.0, 0.1),
            "svc_w": frange(0.0, 2.0, 0.1),
            "dt_w": [1.0]
        },
        # MSDVotingClassifier(): {
        #     "mlp_w": [1.0],
        #     "svc_w": [1.0],
        #     "dt_w": [1.0]
        # }
    }

    scores = dict(zip(learners.keys(), [(-float("inf"), None)] * len(learners.keys())))

    for c in range(3):
        log.info(c)
        for learner, params in learners.items():
            score, best_params = scores[learner]
            gs = GridSearchCV(estimator=learner, param_grid=params, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1)
            gs.fit(x, y)
            if gs.best_score_ > score:
                log.info("{}, {}, {}, {}".format(learner, "Best Score", gs.best_score_, gs.best_params_))
                scores[learner] = (gs.best_score_, gs.best_params_)

    for learner in learners:
        log.info("{} -> {}".format(learner.__class__, scores[learner][0]))

    # base_estimator = DecisionTreeClassifier(max_depth=20, max_features=5, splitter='best')
    # accuracy = []
    # for k in range(5):
    #     print(k)
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    #     learner = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100)
    #     learner.fit(x_train, y_train)
    #     pred = learner.predict(x_test)
    #     accuracy.append(accuracy_score(y_test, pred))
    #
    # print("Acc", sum(accuracy) / len(accuracy))

    # max_features_list = [None, "sqrt", "log2"]
    # max_features_list.extend(list(irange(start=10, end=20, step=2)))

    # ada_boost_est = [
    #     # DecisionTreeClassifier(max_depth=1),
    #     DecisionTreeClassifier(max_depth=5),
    #     # DecisionTreeClassifier(max_depth=10),
    #     # DecisionTreeClassifier(max_depth=20),
    #     KNeighborsClassifier(n_neighbors=2, weights="distance"),
    #     # KNeighborsClassifier(n_neighbors=5, weights="distance"),
    #     # KNeighborsClassifier(n_neighbors=10, weights="distance"),
    #     MLPClassWrapper(hidden_layer_dimension=4, hidden_layer_value=10, learning_rate="invscaling", learning_rate_init=0.009),
    #     # MLPClassWrapper(hidden_layer_dimension=4, hidden_layer_value=20, learning_rate="invscaling", learning_rate_init=0.009),
    #     # MLPClassWrapper(hidden_layer_dimension=6, hidden_layer_value=10, learning_rate="invscaling", learning_rate_init=0.009),
    #     # MLPClassWrapper(hidden_layer_dimension=6, hidden_layer_value=20, learning_rate="invscaling", learning_rate_init=0.009),
    # ]

    # learners = [
    #     (DecisionTreeClassifier(),
    #      {"max_features": max_features_list, "criterion": ("gini", "entropy", "log_loss"), "ccp_alpha": frange(),
    #       "max_depth": irange(1, 100, 10), "splitter": ("best", "random")}),
    #
    #     (KNeighborsClassifier(),
    #      {"n_neighbors": irange(10, 50, 5), "weights": ('uniform', 'distance'), }),
    #
    #     # (XGBClassifier(),
    #     #  {"max_depth": irange(1, 100, 10), "grow_policy": ["depthwise", "lossguide"], "alpha": frange()}),
    #     #
    #     #
    #     # (MLPClassWrapper(),
    #     #  {"hidden_layer_dimension": irange(1, 50, 5), "hidden_layer_value": irange(1, 50, 5),
    #     #   "alpha": frange(0.0, 0.001, 0.0001),
    #     #   "learning_rate": ('constant', 'invscaling', 'adaptive'),
    #     #   "learning_rate_init": frange(0.001, 0.1, 0.01),
    #     #   "max_iter": [1000]
    #     #   }),
    #     #
    #     # (AdaBoostClassifier(),
    #     #  {"n_estimators": irange(1, 100, 5), "learning_rate": frange(0.1, 2.0, 0.1), "base_estimator": ada_boost_est}),
    #     #
    #     # (SVC(), {"C": frange(start=0.1, end=2.0, step=0.2), "gamma": ['scale', 'auto'],
    #     #          "kernel": ['linear', 'poly', 'rbf', 'sigmoid']}),
    # ]
    #
    # for learner, search_param in learners:
    #     gs = GridSearchCV(estimator=learner, param_grid=search_param, scoring="accuracy", cv=5, n_jobs=-1)
    #     gs.fit(x_train, y_train)
    #     print(learner)
    #     print(gs.best_params_)
    #     print(gs.best_score_)
    #     # scores = cross_validate(l, x, y, cv=5, scoring=make_scorer(s, **p), n_jobs=-1)
    #     # mean = np.mean(scores["test_score"])
    #     # print(l, mean)


if __name__ == "__main__":
    main()

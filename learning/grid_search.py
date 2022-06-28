import pickle
from os.path import join
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from learning.MLP import MLPClassWrapper


def frange(start=0, end=1, step=0.1):
    out = [start]
    while out[-1] <= end:
        out.append(out[-1] + step)
    return out


def irange(start=0, end=10, step=1):
    return list(range(start, end, step))


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
        master_df[tile] = pd.qcut(master_df['xwoba'], bins[tile], labels=False)

    x = master_df[['FF_release_spin_rate_mean', 'FF_release_spin_rate_std', 'FF_effective_speed_mean',
                       'FF_effective_speed_std', 'FF_spin_axis_mean', 'FF_spin_axis_std', 'FF_active_spin',
                       'SL_release_spin_rate_mean', 'SL_release_spin_rate_std', 'SL_effective_speed_mean',
                       'SL_effective_speed_std', 'SL_spin_axis_mean', 'SL_spin_axis_std', 'SL_active_spin',
                       'CUKC_release_spin_rate_mean', 'CUKC_release_spin_rate_std', 'CUKC_effective_speed_mean',
                       'CUKC_effective_speed_std', 'CUKC_spin_axis_mean', 'CUKC_spin_axis_std', 'CUKC_active_spin',
                       'CH_release_spin_rate_mean', 'CH_release_spin_rate_std', 'CH_effective_speed_mean',
                       'CH_effective_speed_std', 'CH_spin_axis_mean', 'CH_spin_axis_std', 'CH_active_spin',
                       'SIFT_release_spin_rate_mean', 'SIFT_release_spin_rate_std', 'SIFT_effective_speed_mean',
                       'SIFT_effective_speed_std', 'SIFT_spin_axis_mean', 'SIFT_spin_axis_std', 'SIFT_active_spin',
                       'FC_release_spin_rate_mean', 'FC_release_spin_rate_std', 'FC_effective_speed_mean',
                       'FC_effective_speed_std', 'FC_spin_axis_mean', 'FC_spin_axis_std', 'FC_active_spin']].fillna(0)

    y = master_df['quintile']

    print(len(y))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    learners = [
        (DecisionTreeClassifier(),
         {"max_features": ("sqrt", "log2", None), "criterion": ("gini", "entropy", "log_loss"), "ccp_alpha": frange(),
          "max_depth": irange(1, 100, 10), "splitter": ("best", "random")}),
        (KNeighborsClassifier(), {"n_neighbors": irange(10, 50, 5), "weights": ('uniform', 'distance'), }),
        (MLPClassWrapper(), {"hidden_layer_dimension": irange(1, 50, 5), "hidden_layer_value": irange(1, 50, 5),
                             "alpha": frange(0.0, 0.001, 0.0001),
                             "learning_rate": ('constant', 'invscaling', 'adaptive'),
                             "learning_rate_init": frange(0.001, 0.01, 0.001),
                             "max_iter": [1000]
                             }),

        (SVC(), {"kernel": ['poly'], "degree": irange(start=1, end=10, step=1)}),
        # "decision_tree_class": (xgb.XGBClassifier, precision_score, {"average": 'micro'}),
        # (AdaBoostClassifier(), )
    ]

    for learner, search_param in learners:
        gs = GridSearchCV(estimator=learner, param_grid=search_param, scoring="accuracy", cv=5, n_jobs=-1)
        gs.fit(x_train, y_train)
        print(learner)
        print(gs.best_params_)
        print(gs.best_score_)
        # scores = cross_validate(l, x, y, cv=5, scoring=make_scorer(s, **p), n_jobs=-1)
        # mean = np.mean(scores["test_score"])
        # print(l, mean)


if __name__ == "__main__":
    main()
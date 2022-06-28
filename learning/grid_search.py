import pickle
from os.path import join
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
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
        master_df[tile] = pd.qcut(master_df['xFIP'], bins[tile], labels=False)

    columns = ['FF_effective_speed_mean', 'CUKC_pfx_z_mean', 'CUKC_spin_axis_std', 'CH_spin_axis_mean',
               'FF_pitcher_break_z', 'SIFT_effective_speed_mean', 'FF_effective_speed_std',
               'CH_release_spin_rate_mean', 'CH_pfx_z_mean', 'SIFT_pfx_x_std', 'CUKC_release_speed_mean',
               'FF_pfx_x_mean', 'CUKC_pfx_x_std', 'FC_pitcher_break_z', 'CH_spin_axis_std', 'FF_spin_axis_std',
               'CUKC_spin_axis_mean', 'CUKC_release_spin_rate_std', 'SL_spin_axis_mean', 'SIFT_pfx_x_mean',
               'SL_effective_speed_mean', 'CUKC_effective_speed_mean', 'FC_tail', 'FC_release_spin_rate_std',
               'SL_rise']
    x = pd.DataFrame(PowerTransformer().fit_transform(master_df[columns].fillna(0)), columns=columns)
    y = master_df['quintile']

    print(len(y))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    max_features_list = [None, "sqrt", "log2"]
    max_features_list.extend(list(irange(start=10, end=20, step=2)))

    ada_boost_est = [
        # DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=5),
        # DecisionTreeClassifier(max_depth=10),
        # DecisionTreeClassifier(max_depth=20),
        KNeighborsClassifier(n_neighbors=2, weights="distance"),
        # KNeighborsClassifier(n_neighbors=5, weights="distance"),
        # KNeighborsClassifier(n_neighbors=10, weights="distance"),
        MLPClassWrapper(hidden_layer_dimension=4, hidden_layer_value=10, learning_rate="invscaling", learning_rate_init=0.009),
        # MLPClassWrapper(hidden_layer_dimension=4, hidden_layer_value=20, learning_rate="invscaling", learning_rate_init=0.009),
        # MLPClassWrapper(hidden_layer_dimension=6, hidden_layer_value=10, learning_rate="invscaling", learning_rate_init=0.009),
        # MLPClassWrapper(hidden_layer_dimension=6, hidden_layer_value=20, learning_rate="invscaling", learning_rate_init=0.009),
    ]

    learners = [
        (DecisionTreeClassifier(),
         {"max_features": max_features_list, "criterion": ("gini", "entropy", "log_loss"), "ccp_alpha": frange(),
          "max_depth": irange(1, 100, 10), "splitter": ("best", "random")}),

        (KNeighborsClassifier(),
         {"n_neighbors": irange(10, 50, 5), "weights": ('uniform', 'distance'), }),

        (XGBClassifier(),
         {"max_depth": irange(1, 100, 10), "grow_policy": ["depthwise", "lossguide"], "alpha": frange()}),


        (MLPClassWrapper(),
         {"hidden_layer_dimension": irange(1, 50, 5), "hidden_layer_value": irange(1, 50, 5),
          "alpha": frange(0.0, 0.001, 0.0001),
          "learning_rate": ('constant', 'invscaling', 'adaptive'),
          "learning_rate_init": frange(0.001, 0.1, 0.01),
          "max_iter": [1000]
          }),

        (AdaBoostClassifier(),
         {"n_estimators": irange(1, 100, 5), "learning_rate": frange(0.1, 2.0, 0.1), "base_estimator": ada_boost_est}),

        (SVC(), {"C": frange(start=0.1, end=2.0, step=0.2), "gamma": ['scale', 'auto'],
                 "kernel": ['linear', 'poly', 'rbf', 'sigmoid']}),
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

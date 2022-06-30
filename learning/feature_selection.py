import pickle
from os.path import join

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, f1_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import RobustScaler, Normalizer, MaxAbsScaler, MinMaxScaler, PowerTransformer, \
    QuantileTransformer, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main():
    with open(join("learning", "combined.pkl"), "rb") as f:
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

    suffixed_columns = ["release_spin_rate", "effective_speed", "spin_axis", "release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z"]
    # strike_zone = ["strike_high", "strike_middle", "strike_low", "ball_high", "ball_low"]
    suffixes = ["mean", "std"]
    pitches = ["FF", "SL", "CUKC", "CH", "SIFT", "FC"]

    columns = []
    for pitch in pitches:
        for sc in suffixed_columns:
            for suffix in suffixes:
                columns.append(pitch + "_" + sc + "_" + suffix)
        # for sz in strike_zone:
        #     columns.append(pitch + "_" + sz)

    for c in columns:
        if c not in master_df.columns:
            print("Missing", c)


    x = pd.DataFrame(PowerTransformer().fit_transform(master_df[columns].fillna(0)), columns=columns)
    y = master_df['tercile']

    # learner = DecisionTreeClassifier(criterion='entropy', max_depth=75)
    # scores = cross_validate(learner, x, y, cv=5, scoring=make_scorer(accuracy_score), n_jobs=-1)
    # print(np.mean(scores["test_score"]))

    print(sklearn.metrics.get_scorer_names())

    RFECV_results = []
    for i in range(200):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        selector = RFECV(DecisionTreeClassifier(criterion='entropy', max_depth=75), cv=5, scoring="f1_micro")
        selector = selector.fit(X_train, y_train)
        m = max(selector.cv_results_["mean_test_score"])
        print(i, m)
        RFECV_results.append((m, selector.feature_names_in_[selector.support_]))

    c_results = {}
    for column in columns:
        c_score = 0
        for score, features in RFECV_results:
            if column in features:
                c_score += score
        c_results[column] = c_score / len(RFECV_results)

    marklist = sorted(c_results.items(), key=lambda x: x[1], reverse=True)
    sortdict = dict(marklist)

    total = sum(sortdict.values())

    for key in sortdict:
        sortdict[key] = sortdict[key] / total

    print(sortdict)

    all_scores = []
    best_score = -float("inf")
    best_idx = -1

    s_columns = []
    for i, key in enumerate(sortdict):
        s_columns.append(key)
        total_score = []
        for _ in range(50):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
            learner = DecisionTreeClassifier(criterion='entropy', max_depth=75)
            learner.fit(X_train[s_columns], y_train)
            total_score.append(f1_score(y_test, learner.predict(X_test[s_columns]), average="micro"))
        avg_score = sum(total_score) / len(total_score)
        all_scores.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            best_idx = i

    improvement = [0]

    for i in range(1, len(all_scores)):
        avg_score = sum(all_scores[:i]) / i
        improvement.append(all_scores[i] - avg_score)

    print("Bests", best_score, best_idx)
    print("Best Cols", s_columns[:best_idx + 1])
    print("col", s_columns)
    print("mse", avg_score)
    print("improvement", improvement)

    # for i in range(1, len(x.columns) + 1):
    #     pca = PCA(n_components=i)
    #     nx = pca.fit_transform(x)
    #     learner = MLPClassWrapper(hidden_layer_dimension=6, hidden_layer_value=36, learning_rate='invscaling', learning_rate_init=0.009)
    #     scores = cross_validate(learner, nx, y, cv=5, scoring=make_scorer(accuracy_score), n_jobs=-1)
    #     print(i, np.mean(scores["test_score"]))


if __name__ == "__main__":
    main()
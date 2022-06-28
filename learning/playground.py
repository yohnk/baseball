import pickle
from os.path import join

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

from learning.MLP import MLPClassWrapper


def main():
    with open(join("learning", "combined.pkl"), "rb") as f:
        master_df = pickle.load(f)
        print(master_df.columns)

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

    x = master_df[['FF_release_spin_rate_mean', 'FF_effective_speed_mean', 'FF_spin_axis_mean', 'FF_active_spin',
                   "FF_zones",
                   # 'FF_strike_high', 'FF_strike_middle', 'FF_strike_low', 'FF_ball_high', 'FF_ball_low',
                   'SL_release_spin_rate_mean', 'SL_effective_speed_mean', 'SL_spin_axis_mean', 'SL_active_spin',
                   "SL_zones",
                   # 'SL_strike_high', 'SL_strike_middle', 'SL_strike_low', 'SL_ball_high', 'SL_ball_low',
                   'CUKC_release_spin_rate_mean', 'CUKC_effective_speed_mean', 'CUKC_spin_axis_mean', 'CUKC_active_spin',
                   "CUKC_zones",
                   # 'CUKC_strike_high', 'CUKC_strike_middle', 'CUKC_strike_low', 'CUKC_ball_high', 'CUKC_ball_low',
                   'CH_release_spin_rate_mean', 'CH_effective_speed_mean', 'CH_spin_axis_mean', 'CH_active_spin',
                   "CH_zones",
                   # 'CH_strike_high', 'CH_strike_middle', 'CH_strike_low', 'CH_ball_high', 'CH_ball_low',
                   'SIFT_release_spin_rate_mean', 'SIFT_effective_speed_mean', 'SIFT_spin_axis_mean', 'SIFT_active_spin',
                   "SIFT_zones",
                   # 'SIFT_strike_high', 'SIFT_strike_middle', 'SIFT_strike_low', 'SIFT_ball_high', 'SIFT_ball_low',
                   'FC_release_spin_rate_mean', 'FC_effective_speed_mean', 'FC_spin_axis_mean', 'FC_active_spin',
                   "FC_zones"
                   # 'FC_strike_high', 'FC_strike_middle', 'FC_strike_low', 'FC_ball_high', 'FC_ball_low'
                   ]].fillna(0)

    y = master_df['docile']

    learner = DecisionTreeClassifier(criterion='entropy', max_depth=75)
    scores = cross_validate(learner, x, y, cv=5, scoring=make_scorer(accuracy_score), n_jobs=-1)
    print(np.mean(scores["test_score"]))

    selector = RFECV(DecisionTreeClassifier(criterion='entropy', max_depth=75), cv=5)
    selector = selector.fit(x, y)
    print(max(selector.cv_results_["mean_test_score"]))

    # for i in range(1, len(x.columns) + 1):
    #     pca = PCA(n_components=i)
    #     nx = pca.fit_transform(x)
    #     learner = MLPClassWrapper(hidden_layer_dimension=6, hidden_layer_value=36, learning_rate='invscaling', learning_rate_init=0.009)
    #     scores = cross_validate(learner, nx, y, cv=5, scoring=make_scorer(accuracy_score), n_jobs=-1)
    #     print(i, np.mean(scores["test_score"]))


if __name__ == "__main__":
    main()
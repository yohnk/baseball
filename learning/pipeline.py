import pickle
from os.path import join

import numpy as np
import pandas as pd
import pybaseball
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier

from learning.custom_learners import VotingAdaBoost, DTAdaBoost, MSDVotingClassifier, SVCAdaBoost


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

    train = master_df[master_df.year != 2022]

    x_train = pd.DataFrame(PowerTransformer().fit_transform(train[columns].fillna(0)), columns=columns)
    y_train = train['tercile']

    vab = VotingAdaBoost()
    vab.fit(x_train, y_train)

    regressors = {}
    for tercile in pd.unique(train["tercile"]):
        n_train = train[(train.quintile <= (tercile * 2) + 1) & (train.quintile >= (tercile * 2) - 1)]
        # n_test = test[(test.quintile <= (tercile * 2) + 1) & (test.quintile >= (tercile * 2) - 1)]

        x_train = pd.DataFrame(PowerTransformer().fit_transform(n_train[columns].fillna(0)), columns=columns)
        y_train = n_train['xFIP']
        # x_test = pd.DataFrame(PowerTransformer().fit_transform(n_test[columns].fillna(0)), columns=columns)
        # y_test = n_test['xFIP']

        svr = SVR()
        svr.fit(x_train, y_train)
        regressors[tercile] = svr

    test = master_df[master_df.year == 2022]
    test['pred_tercile'] = vab.predict(PowerTransformer().fit_transform(test[columns].fillna(0)))

    for tercile, regressor in regressors.items():
        test.loc[test.pred_tercile == tercile, "pred_xFIP"] = regressor.predict(PowerTransformer().fit_transform(test.loc[test.pred_tercile == tercile][columns].fillna(0)))

    test["xFIP"] = test["xFIP"]
    test["tercile"] = test["tercile"]
    master_df[["tercile", "pred_tercile", "xFIP", "pred_xFIP"]] = test[["tercile", "pred_tercile", "xFIP", "pred_xFIP"]]
    master_df["diff"] = master_df["xFIP"] - master_df["pred_xFIP"]

    final = pd.merge(master_df, pybaseball.chadwick_register(), left_on=['player_id'], right_on=['key_mlbam'])
    final[["year", "player_id", "name_first", "name_last", "tercile", "pred_tercile", "pred_xFIP", "xFIP", "diff"]].dropna(subset=["pred_xFIP"]).to_csv("final.csv")


    # x = pd.DataFrame(PowerTransformer().fit_transform(master_df[columns].fillna(0)), columns=columns)
    # y = master_df['quintile']
    #
    #


if __name__ == "__main__":
    main()

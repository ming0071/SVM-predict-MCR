# -*- coding: utf-8 -*-

import os
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import train_test_split

import module.handle_parm as hd_parm
import module.SVM_model as SVM


def main():

    # Load dataset
    MCR_excel = "MMH-MM-MERGED-0909.xlsx"
    df = pd.read_excel("./dataset/" + MCR_excel)

    # Preprocess data
    label = [0, 1]
    label_name = "認知退化"
    all_feature_mode = True  # True: select all features; False: only select best feature(SELECT_FEATURES)

    # feature selection
    df = hd_parm.add_Class_column(df, label, label_name)
    if all_feature_mode is True:
        # use linear kernel to find best feature(SELECT_FEATURES)
        df = hd_parm.delete_unnecessary_data(df, max_allowed_nan_per_row=20)
        df = hd_parm.impute_nan(
            df, method="bayesian_ridge", max_allowed_nan_per_column=300
        )
        df = hd_parm.process_disease_features(df)
    else:
        # only select best feature(SELECT_FEATURES) to avoid overfitting
        df = hd_parm.process_disease_features(df)
        indices = list(range(0, 60))  # 索引從 0 開始。 init 60
        print(indices)
        df = hd_parm.select_features(df, indices)
        df = hd_parm.clean_dataframe(df, max_allowed_nan_per_row=15)
        df = hd_parm.impute_nan(
            df, method="bayesian_ridge", max_allowed_nan_per_column=300
        )
    df.to_excel("./output/handleparm.xlsx", encoding="utf_8_sig")
    df_df = pd.DataFrame(df.drop(columns=["Class"]))
    feature_names = df_df.columns

    # Count missing feature values
    Missing_df = hd_parm.count_missing_values(df)
    Missing_df.to_excel("./output/missing.xlsx", encoding="utf_8_sig")

    # Feature selection
    PCA = "None"
    Y = df["Class"].values
    df = df.drop(columns=["Class"])
    df = hd_parm.normalize_data(df)
    df = hd_parm.perform_pca(df, PCA)

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        df, Y, test_size=0.2, random_state=90
    )
    print("Original :")
    print("Y_train  : {}".format(Counter(y_train)))
    print("Y_test   : {}".format(Counter(y_test)))

    # Oversample training data
    sm = SVMSMOTE(random_state=71)  # 71
    x_train, y_train = sm.fit_resample(x_train, y_train)
    print("Resample :")
    print("Y_train  : {}".format(Counter(y_train)))
    print("Y_test   : {}".format(Counter(y_test)))

    # Compare SVM models
    kernel = ["linear", "rbf", "poly"]
    svm_model = SVM.SVMModel()

    predict_table, svcModel = svm_model.compare_SVM(
        feature_names, kernel, x_train, x_test, y_train, y_test, label
    )

    # Save final results
    predict_df = pd.DataFrame(predict_table)
    predict_df.to_excel(
        "./output/SVM predict.xlsx", encoding="utf_8_sig", sheet_name="PCA" + str(PCA)
    )


if __name__ == "__main__":
    os.system("cls")
    main()

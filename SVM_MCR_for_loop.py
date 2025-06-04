# -*- coding: utf-8 -*-
import os
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import train_test_split

import module.handle_parm as hd_parm
import module.SVM_model as SVM


def main():
    # 讀取資料
    MCR_excel = "MMH-MM-MERGED-0909.xlsx"
    df_original = pd.read_excel("./dataset/" + MCR_excel)

    # 資料預處理：新增 Class 欄位等
    label = [0, 1]
    label_name = "認知退化"
    # 建議對所有資料先執行與疾病相關的特徵處理
    df_original = hd_parm.add_Class_column(df_original, label, label_name)

    # 建立最終結果列表
    final_result_table = []  # 每筆結果為一 dict，後續轉為 DataFrame

    # 迴圈：從 5 開始，每次增加 1 個特徵，直到 61 個  (5, 62, 1)
    for feature_num in range(28, 36, 1):
        # 每次從原始資料開始重新處理
        df_temp = hd_parm.process_disease_features(df_original.copy())
        # 選取指定索引範圍（例如：0~feature_num-1）
        indices = list(range(0, feature_num))
        df_temp = hd_parm.select_features(df_temp, indices)
        df_temp = hd_parm.clean_dataframe(df_temp, max_allowed_nan_per_row=15)
        df_temp = hd_parm.impute_nan(
            df_temp, method="bayesian_ridge", max_allowed_nan_per_column=300
        )

        # 儲存處理後的資料（可選，方便後續檢查）
        df_temp.to_excel("./output/handleparm.xlsx", encoding="utf_8_sig")

        # 取出特徵名稱（除去 Class 欄位）
        df_features = pd.DataFrame(df_temp.drop(columns=["Class"]))
        feature_names = df_features.columns

        # 計算缺失值統計（可選）
        Missing_df = hd_parm.count_missing_values(df_temp)
        Missing_df.to_excel("./output/missing.xlsx", encoding="utf_8_sig")

        # 特徵前處理：標準化、PCA（此處 PCA 參數可調整，若無 PCA 則保留原始）
        PCA = "None"
        Y = df_temp["Class"].values
        df_temp = df_temp.drop(columns=["Class"])
        df_temp = hd_parm.normalize_data(df_temp)
        df_temp = hd_parm.perform_pca(df_temp, PCA)

        # 分割資料
        x_train, x_test, y_train, y_test = train_test_split(
            df_temp, Y, test_size=0.2, random_state=90
        )
        print(f"【特徵數量: {feature_num}】")
        print("Original :")
        print("Y_train  : {}".format(Counter(y_train)))
        print("Y_test   : {}".format(Counter(y_test)))

        # 過採樣訓練資料
        sm = SVMSMOTE(random_state=71)
        x_train, y_train = sm.fit_resample(x_train, y_train)
        print("Resample :")
        print("Y_train  : {}".format(Counter(y_train)))
        print("Y_test   : {}".format(Counter(y_test)))

        # 使用不同核函數比較 SVM 模型
        kernel_list = ["linear", "rbf", "poly"]
        svm_model = SVM.SVMModel()
        # compare_SVM 回傳 predict_table 為 list，每個元素包含結果欄位
        predict_table, svcModel = svm_model.compare_SVM(
            feature_names, kernel_list, x_train, x_test, y_train, y_test, label
        )

        # 將本輪的特徵數量資訊加入每筆結果
        final_result_table.extend(predict_table)

    # 將最終結果存成 Excel，欄位順序可依需求排序
    final_df = pd.DataFrame(final_result_table)
    # 假設最終結果中已有欄位: kernel, Train ACC, Test ACC, AUC, corss ACC, corss AUC
    # 如需重新排列，可用 final_df = final_df[['kernel', 'Feature_num', 'Train ACC', 'Test ACC', 'AUC', 'corss ACC', 'corss AUC']]
    final_df.to_excel(
        "./output/SVM predict.xlsx", encoding="utf_8_sig", sheet_name="PCA" + str(PCA)
    )


if __name__ == "__main__":
    os.system("cls")
    main()

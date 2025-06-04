import os
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV

import module.handle_parm as hd_parm


def main():
    # 讀取資料
    MCR_excel = "MMH-MM-MERGED-0909.xlsx"
    df_original = pd.read_excel("./dataset/" + MCR_excel)

    # 新增 Class 欄位
    label = [0, 1]
    label_name = "認知退化"
    df_original = hd_parm.add_Class_column(df_original, label, label_name)

    # 結果儲存表格
    results = []

    # 迴圈測試不同特徵數量
    for feature_num in range(5, 62, 1):
        df_temp = hd_parm.process_disease_features(df_original.copy())
        indices = list(range(0, feature_num))
        df_temp = hd_parm.select_features(df_temp, indices)
        df_temp = hd_parm.clean_dataframe(df_temp, max_allowed_nan_per_row=15)
        df_temp = hd_parm.impute_nan(
            df_temp, method="bayesian_ridge", max_allowed_nan_per_column=300
        )

        print(f"【特徵數量: {feature_num}】")
        X = df_temp.drop(columns=["Class"])
        y = df_temp["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        smote = SVMSMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Grid Search 範圍設定
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 10, None],
        }

        rf_clf = RandomForestClassifier(
            class_weight="balanced",
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        )

        grid_search = GridSearchCV(
            rf_clf,
            param_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_resampled, y_resampled)
        best_params = grid_search.best_params_
        best_rf_model = grid_search.best_estimator_

        # 留出法預測
        y_pred = best_rf_model.predict(X_test)
        y_proba = best_rf_model.predict_proba(X_test)[:, 1]

        train_acc = best_rf_model.score(X_resampled, y_resampled)
        test_acc = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)

        # K-fold 驗證
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        fold_aucs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train_k, X_test_k = X.iloc[train_idx], X.iloc[test_idx]
            y_train_k, y_test_k = y.iloc[train_idx], y.iloc[test_idx]
            X_res_k, y_res_k = smote.fit_resample(X_train_k, y_train_k)

            rf_model_k = RandomForestClassifier(
                **best_params,
                class_weight="balanced",
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            )
            rf_model_k.fit(X_res_k, y_res_k)

            y_pred_k = rf_model_k.predict(X_test_k)
            y_proba_k = rf_model_k.predict_proba(X_test_k)[:, 1]

            fold_accs.append(accuracy_score(y_test_k, y_pred_k))
            fold_aucs.append(roc_auc_score(y_test_k, y_proba_k))

        # 儲存結果
        results.append(
            {
                "Feature_num": feature_num,
                "Train ACC": train_acc,
                "Test ACC": test_acc,
                "Test AUC": test_auc,
                **{f"{i+1} fold ACC": acc for i, acc in enumerate(fold_accs)},
                **{f"{i+1} fold AUC": auc for i, auc in enumerate(fold_aucs)},
            }
        )

    # 輸出為 Excel
    df_results = pd.DataFrame(results)
    df_results.to_excel(
        "./output/RF predict.xlsx",
        index=False,
        encoding="utf_8_sig",
    )


if __name__ == "__main__":
    main()

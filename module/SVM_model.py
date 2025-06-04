import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV

import module.save_result as save_result
import module.plot_result_curve as plt_result


class SVMModel:
    def __init__(self):
        # Define hyperparameters for different SVM kernels
        self.parameters = {
            "linear": {
                "C": np.logspace(-1, 3, num=20),
            },
            "rbf": {
                "C": np.logspace(-1, 3, num=20),
                "gamma": ["auto", "scale", *np.logspace(-4, 2, num=30)],
            },
            "poly": {
                "C": np.logspace(-1, 3, num=10),
                "degree": [2, 3, 4],
                "gamma": ["auto", "scale", *np.logspace(-4, 2, num=20)],
            },
            "sigmoid": {
                "C": np.logspace(-1, 3, num=20),
                "gamma": ["auto", "scale", *np.logspace(-4, 2, num=30)],
            },
        }

    def grid_search(self, kernel, X_train, y_train):
        # Select the parameter grid based on the kernel type
        param_grid = self.parameters[kernel]

        # Initialize GridSearchCV to perform hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=SVC(),
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            return_train_score=True,
            n_jobs=-1,  # Enable parallel processing
        )

        # Fit the GridSearchCV to the training data
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters and estimator
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_

        print("GridSearch CV :")
        print(
            "best c : {} , best gamma : {} ".format(
                best_params.get("C", "auto"), best_params.get("gamma", "scale")
            )
        )
        print("Best train score : {}".format(grid_search.best_score_))

        return best_params, best_estimator

    def kfold(self, model, k, df_x, df_y):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        # 一次同時計算多個 scoring
        scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]
        results = cross_validate(model, df_x, df_y, cv=skf, scoring=scoring)

        df_results = pd.DataFrame(
            {
                "Accuracy": results["test_accuracy"],
                "F1 Score": results["test_f1_macro"],
                "Precision": results["test_precision_macro"],
                "Recall": results["test_recall_macro"],
                "AUC": results["test_roc_auc"],
            }
        )

        print("\n{}-fold Cross Validation results:\n".format(k))
        print(df_results)

        # 平均值計算
        mean_f1 = np.mean(results["test_f1_macro"])
        mean_precision = np.mean(results["test_precision_macro"])
        mean_recall = np.mean(results["test_recall_macro"])
        # mean_acc = np.mean(results["test_accuracy"])
        # mean_auc = np.mean(results["test_roc_auc"])

        return (
            results["test_accuracy"],
            results["test_roc_auc"],
            mean_f1,
            mean_precision,
            mean_recall,
        )

    def get_feature_importance(self, model, X, y, kernel, feature_names):
        if kernel == "linear":
            # For linear kernel, coefficients indicate feature importance
            importance = model.coef_.flatten()
            feature_importance = pd.DataFrame(
                {"Feature": feature_names, "Importance": importance}
            )
        else:
            # For non-linear kernels, use permutation importance
            result = permutation_importance(
                model, X, y, n_repeats=30, random_state=0, n_jobs=-1
            )
            feature_importance = pd.DataFrame(
                {"Feature": feature_names, "Importance": result.importances_mean}
            )

        # Sort features by importance and reset the index
        feature_importance = feature_importance.reindex(
            feature_importance["Importance"].abs().sort_values(ascending=False).index
        )
        feature_importance.reset_index(drop=True, inplace=True)
        return feature_importance

    def compare_SVM(
        self, feature_names, kernels, x_train, x_test, y_train, y_test, label
    ):
        final_result_table = []
        # Compare SVM models with different kernels
        for kernel in kernels:
            print("-----------------------------------------------")
            print("---               kernel :{}              ---".format(kernel))
            print("-----------------------------------------------\n")

            # Perform grid search to find the best parameters and estimator
            best_params, best_estimator = self.grid_search(kernel, x_train, y_train)

            # Create and train the SVM model with the best parameters
            svcModel = SVC(
                kernel=kernel,
                C=best_params.get("C", 1.0),
                max_iter=3000,
                degree=best_params.get("degree", 3),
                gamma=best_params.get("gamma", "scale"),
            )

            svcModel.fit(x_train, y_train)
            train_acc = svcModel.score(x_train, y_train)
            test_acc = svcModel.score(x_test, y_test)
            print("Train ACC : {} , Test ACC : {}".format(train_acc, test_acc))

            # Predict and plot the confusion matrix
            y_pred = svcModel.predict(x_test)
            # self._plot_confusion_mat(y_test, y_pred, label, kernel)

            # precision = precision_score(y_test, y_pred, average="binary")
            # recall = recall_score(y_test, y_pred, average="binary")
            # f1 = f1_score(y_test, y_pred, average="binary")

            # Plot ROC curve and calculate AUC
            AUC, cross_AUC = plt_result.plot_roc_curve_and_auc(
                kernel, svcModel, x_train, y_train, y_test, y_pred
            )

            # Combine train and test data for K-fold cross-validation
            X = np.concatenate((x_train, x_test), axis=0)
            Y = np.concatenate((y_train, y_test), axis=0)
            (
                cross_ACC_list,
                cross_AUC_list,
                mean_f1,
                mean_precision,
                mean_recall,
            ) = self.kfold(svcModel, 5, X, Y)

            # Save results and update the final result
            save_result.report(y_test, y_pred)
            result = save_result.final(
                kernel,
                x_train.shape[1],
                round(train_acc, 3),
                round(test_acc, 3),
                round(AUC, 3),
                round(cross_ACC_list[0], 3),
                round(cross_ACC_list[1], 3),
                round(cross_ACC_list[2], 3),
                round(cross_ACC_list[3], 3),
                round(cross_ACC_list[4], 3),
                round(cross_AUC_list[0], 3),
                round(cross_AUC_list[1], 3),
                round(cross_AUC_list[2], 3),
                round(cross_AUC_list[3], 3),
                round(cross_AUC_list[4], 3),
                round(mean_f1, 3),
                round(mean_precision, 3),
                round(mean_recall, 3),
            )
            final_result_table.append(result)

            # Get feature importance
            feature_importance = self.get_feature_importance(
                svcModel, x_train, y_train, kernel, feature_names
            )
            print("\nFeature importance for kernel {}:\n".format(kernel))
            print(feature_importance)
            filename = "./output/feature importance {}.xlsx".format(kernel)
            feature_importance.to_excel(filename, encoding="utf_8_sig")

        return final_result_table, svcModel

    def _plot_confusion_mat(self, y_test, y_pred, labels, kernel):
        # Generate and print the confusion matrix
        confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
        print("\nConfusion matrix : \n{}".format(confusion_mat))

        # Convert the confusion matrix to a DataFrame and plot it using seaborn
        cm_matrix = pd.DataFrame(
            data=confusion_mat,
            columns=["Predict Positive:1", "Predict Negative:0"],
            index=["Actual Positive:1", "Actual Negative:0"],
        )
        plt.figure()
        plt.title("Confusion Matrix of SVM model with {}".format(kernel))
        sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="YlGnBu")
        plt.show()

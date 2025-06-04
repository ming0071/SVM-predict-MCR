# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, validation_curve

# https://www.kaggle.com/code/prashant111/svm-classifier-tutorial/notebook#18.-ROC---AUC-


# 定義一個函數來計算ROC曲線和AUC值
def plot_roc_curve_and_auc(model_name, model, x_train, y_train, y_test, y_pred):

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    ROC_AUC = roc_auc_score(y_test, y_pred)
    print("ROC AUC : {:.4f}".format(ROC_AUC))

    # plt.figure()
    # plt.plot(fpr, tpr, linewidth=2, label="AUC = %0.2f" % ROC_AUC)
    # plt.plot([0, 1], [0, 1], "k--")
    # plt.rcParams["font.size"] = 12
    # plt.title("ROC Curve of {}".format(model_name))
    # plt.xlabel("False Positive Rate (1 - Specificity)")
    # plt.ylabel("True Positive Rate (Sensitivity)")
    # plt.show()

    Cross_validated_ROC_AUC = cross_val_score(
        model, x_train, y_train, cv=5, scoring="roc_auc"
    ).mean()
    print("Cross validated ROC AUC : {:.4f}".format(Cross_validated_ROC_AUC))

    return ROC_AUC, Cross_validated_ROC_AUC


def plot_validation_curve(estimator, X, y, param_name, param_range):
    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        scoring="accuracy",
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(
        param_range,
        train_scores_mean,
        label="Training accuracy",
        color="darkorange",
        lw=lw,
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.semilogx(
        param_range,
        test_scores_mean,
        label="Test accuracy",
        color="navy",
        lw=lw,
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.show()

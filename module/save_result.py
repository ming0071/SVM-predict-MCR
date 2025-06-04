# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:46:55 2021

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import warnings

# Initialize results
result = {
    "kernel": [],
    "Feature_num": [],
    "Train ACC": [],
    "Test ACC": [],
    "AUC": [],
    "1 folder ACC": [],
    "2 folder ACC": [],
    "3 folder ACC": [],
    "4 folder ACC": [],
    "5 folder ACC": [],
    "1 folder AUC": [],
    "2 folder AUC": [],
    "3 folder AUC": [],
    "4 folder AUC": [],
    "5 folder AUC": [],
    "mean precision": [],
    "mean recall": [],
    "mean f1": [],
}


def final(
    kernel,
    Feature_num,
    Train_acc,
    Test_acc,
    AUC,
    folder_1_ACC,
    folder_2_ACC,
    folder_3_ACC,
    folder_4_ACC,
    folder_5_ACC,
    folder_1_AUC,
    folder_2_AUC,
    folder_3_AUC,
    folder_4_AUC,
    folder_5_AUC,
    precision,
    recall,
    f1,
):
    return {
        "kernel": kernel,
        "Feature_num": Feature_num,
        "Train ACC": Train_acc,
        "Test ACC": Test_acc,
        "AUC": AUC,
        "1 folder ACC": folder_1_ACC,
        "2 folder ACC": folder_2_ACC,
        "3 folder ACC": folder_3_ACC,
        "4 folder ACC": folder_4_ACC,
        "5 folder ACC": folder_5_ACC,
        "1 folder AUC": folder_1_AUC,
        "2 folder AUC": folder_2_AUC,
        "3 folder AUC": folder_3_AUC,
        "4 folder AUC": folder_4_AUC,
        "5 folder AUC": folder_5_AUC,
        "mean precision": precision,
        "mean recall": recall,
        "mean f1": f1,
    }


def report(y_test, y_pred):

    warnings.filterwarnings("ignore")
    report = classification_report(y_test, y_pred)
    print(report)

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Literal
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge


# Constants
LABEL_MAPPING = {
    1: 0,
    "步態慢無失智": 0,
    "VD": 0,
    2: 1,
    "小中風過": 1,
    "MCR": 1,
    "CDR:0.5": 1,
    "MCI": 2,
    "MCI步態慢": 2,
    "AD MCI": 2,
    "mci or 1": 2,
    3: 3,
    "帕金森個案": 3,
}

UNNECESSARY_COLUMNS = [
    "受試者姓名",
    "受試者編號",
    "病歷號",
    "神經心理功能測驗日期",
    "出生年月日",
    "其它補述",
    "分組",
    "身高",
    "體重",
    "職業",
    "現在仍有工作",
    "是否fall ",
    "失智症",
    "主觀記憶",
    "FRIED總分",
    "教育程度",
    "是否有受傷",
    "受傷部位",
    "JUMPS_FlightHeight",
    "JUMPS_COMHeight",
    "alpha-synuclein Mean",  # 包含30個特徵屬於另外的神經資料
    "TUG_MidTurn",
    "錯誤數2",
    "Right Stride Duration (s)",
    "Left Stride Duration (s)",
    "TUG_SitToStand",
    "正確數2",
    "TUG_Forward",
    "Right Stride Duration (s)",
    "錯誤數",
    "算數總數",
    "TUG_StandToSit",
    "扣子顆數",
    "癌症",
    "symmetry index",
    "TUG_Return",
    "Left Stride Duration (s)",
    "算術總數2",
    "TUG_扣子顆數",
    "正確數",
    "TUG(s)",
    "其他",
    "TUG_EndTurn",
    "Right Stride Duration (s)",
    "symmetry index_m",
    "Left Stride Duration (s)",
    "symmetry index_c",
    "TUG_手機計時",
    "氣喘",
]

INVALID_VALUES = [
    "",
    "#DIV/0!",
    "#VALUE!",
    "未做測驗",
    "錯誤",
    "不適用",
    "拄拐杖無法測驗",
]

DISEASE_FEATURES = [
    "高血壓",
    "有無疾病",
    "糖尿病",
    "視覺疾病",
    "心臟疾病",
    # "氣喘",
    "神經系統疾病",
    # "癌症",
    # "其他",
]
FOOTSTEPS_UNNECESSARY_FEATURES = [
    "speedSD",
    "cadenceSD",
    "stride lengthSD",
    "stride length V",
    "Left Stride DurationSD",
    "Left stride duration V",
    "Right Stride DurationSD",
    "Right stride duration V",
    "speed_mSD",
    "cadence_mSD",
    "stride length_mSD",
    "stride length V.1",
    "Left Stride Duration_mSD",
    "Left stride duration V.1",
    "Right Stride Duration_mSD",
    "Right stride duration V.1",
    "speed_cSD",
    "cadence_cSD",
    "stride length_cSD",
    "stride length V.2",
    "Left Stride Duration_cSD",
    "Left stride duration V.2",
    "Right Stride Duration_cSD",
    "Right stride duration V.2",
]


SVM_SELECT_FEATURES = [
    "線條方向",
    "TUG_StandToSit_c",
    "GDS-15",
    "cadence(step/min)",
    "Abeta1-42 Mean",
    "視覺疾病",
    "TUG_SitToStand_m",
    "cadence_c",
    "Tau Mean",
    "CVLT-SF-cued",
    "握力左",
    "年齡",
    "TMT-B正確數",
    "CVLT-SF-30S",
    "CVLT-SF",
    "BMI",
    "TUG_StandToSit_m",
    "教育年數",
    "心臟疾病",
    "Left Stride Duration_c",
    "TMT-A",
    "TUG_Return_c",
    "TUG_MidTurn_m",
    "fall次數",
    "TUG_Forward_c",
    "Right Stride Duration_c",
    "CVLT-SF-10MIN",
    "性別",
    "MMSE",
    "波士頓命名(正確+提示後正確)",
    "高血壓",
    "握力右",
    "神經系統疾病",
    "TUG_MidTurn_c",
    "speed(m/s)",
    "TMT-B",
    "疾病加總",
    "TUG_Forward_m",
    "波士頓命名",
    "MOCA",
    "TUG_EndTurn_m",
    "平衡測試",
    "糖尿病",
    "TUG(s)_c",
    "Right Stride Duration_m",
    "五公尺走路",
    "cadence_m",
    "FRIED衰弱期別",
    "stride length_m",
    "TUG_Return_m",
    "speed_c",
    "stride length_c",
    "TUG(s)_m",
    "TUG_EndTurn_c",
    "Left Stride Duration_m",
    "數字符號",
    "stride length(m)",
    "五下坐站",
    "speed_m",
    "TMT-A 正確數",
    "TUG_SitToStand_c",
]


RF_SELECT_FEATURES = [
    "CVLT-SF-30S",
    "CVLT-SF-cued",
    "CVLT-SF",
    "五下坐站",
    "CVLT-SF-10MIN",
    "波士頓命名",
    "Tau Mean",
    "TUG_SitToStand_m",
    "視覺疾病",
    "TMT-B",
    "MMSE",
    "線條方向",
    "MOCA",
    "數字符號",
    "握力右",
    "TUG_Forward_m",
    "TUG_EndTurn_m",
    "波士頓命名(正確+提示後正確)",
    "性別",
    "Right Stride Duration_c",
    "BMI",
    "TMT-B正確數",
    "年齡",
    "cadence_m",
    "cadence(step/min)",
    "平衡測試",
    "TUG_MidTurn_m",
    "Left Stride Duration_c",
    "cadence_c",
    "疾病加總",
    "Abeta1-42 Mean",
    "TUG_StandToSit_c",
    "TUG_Forward_c",
    "speed_m",
    "握力左",
    "TUG_StandToSit_m",
    "TUG(s)_c",
    "TUG_SitToStand_c",
    "GDS-15",
    "stride length_m",
    "Right Stride Duration_m",
    "TMT-A",
    "stride length(m)",
    "FRIED衰弱期別",
    "speed(m/s)",
    "TUG_Return_c",
    "教育年數",
    "speed_c",
    "TUG(s)_m",
    "Left Stride Duration_m",
    "TUG_EndTurn_c",
    "TUG_Return_m",
    "fall次數",
    "stride length_c",
    "TUG_MidTurn_c",
    "糖尿病",
    "五公尺走路",
    "TMT-A 正確數",
    "高血壓",
    "心臟疾病",
    "神經系統疾病",
]


SVM_RF_SELECT_FEATURES = [
    "CVLT-SF-30S",
    "CVLT-SF-cued",
    "線條方向",
    "CVLT-SF",
    "Tau Mean",
    "視覺疾病",
    "TUG_SitToStand_m",
    "TUG_StandToSit_c",
    "GDS-15",
    "CVLT-SF-10MIN",
    "cadence(step/min)",
    "Abeta1-42 Mean",
    "cadence_c",
    "TMT-B正確數",
    "年齡",
    "BMI",
    "握力左",
    "五下坐站",
    "MMSE",
    "Left Stride Duration_c",
    "TUG_StandToSit_m",
    "TUG_MidTurn_m",
    "Right Stride Duration_c",
    "波士頓命名(正確+提示後正確)",
    "性別",
    "教育年數",
    "握力右",
    "TMT-A",
    "TUG_Forward_c",
    "TUG_Return_c",
    "波士頓命名",
    "TMT-B",
    "心臟疾病",
    "fall次數",
    "MOCA",
    "TUG_Forward_m",
    "TUG_EndTurn_m",
    "TUG_MidTurn_c",
    "高血壓",
    "疾病加總",
    "神經系統疾病",
    "speed(m/s)",
    "平衡測試",
    "數字符號",
    "cadence_m",
    "TUG(s)_c",
    "Right Stride Duration_m",
    "stride length_m",
    "FRIED衰弱期別",
    "糖尿病",
    "speed_m",
    "speed_c",
    "TUG_Return_m",
    "TUG(s)_m",
    "stride length(m)",
    "五公尺走路",
    "TUG_SitToStand_c",
    "stride length_c",
    "TUG_EndTurn_c",
    "Left Stride Duration_m",
    "TMT-A 正確數",
]


# Function to replace the labels in the "分組" column with numerical values
def replace_labels(df):  # 0: normal, 1: MCR, 2: MCI, 3: AD
    df["分組"].replace(LABEL_MAPPING, inplace=True)
    # Remove rows with labels 0 and 3, and drop rows with NaN in "分組" column
    df = df.drop(df[df["分組"].isin([0, 3])].index)
    df.dropna(subset=["分組"], inplace=True)
    return df


# Function to add a "Class" column based on the specified label and column
def add_Class_column(df, labels, col):
    df["Class"] = df[col].replace({0: labels[0], 1: labels[1]})
    df.reset_index(drop=True, inplace=True)  # Reset the index
    return df


def clean_dataframe(df, max_allowed_nan_per_row=10):
    # Replace specific values with NaN
    df.replace(INVALID_VALUES, np.nan, inplace=True)
    # Drop rows with more than max_allowed_nan_per_row NaN values
    df.dropna(thresh=df.shape[1] - max_allowed_nan_per_row, inplace=True)
    # Drop the specified column and reset the index
    df.drop(columns=["認知退化"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# Function to delete unwanted columns and clean the data
def delete_unnecessary_data(df, max_allowed_nan_per_row):
    # Drop columns that are unnecessary or have incomplete data
    df.drop(columns=UNNECESSARY_COLUMNS + FOOTSTEPS_UNNECESSARY_FEATURES, inplace=True)

    df = clean_dataframe(df, max_allowed_nan_per_row)
    return df


# Function to count missing values in the dataset
def count_missing_values(data, include_all=False):
    # missing = (data == "NaN").sum()
    missing = (data.isna()).sum()
    if not include_all:
        missing = missing[missing > 0]
    missing.sort_values(ascending=False, inplace=True)
    missing_count = pd.DataFrame(
        {"Column Name": missing.index, "Missing Count": missing.values}
    )
    missing_count["Percentage(%)"] = missing_count["Missing Count"].apply(
        lambda x: "{:.2%}".format(x / data.shape[0])
    )
    return missing_count


# Placeholder function for replacing NaN values (currently does nothing)
def impute_nan(
    df: pd.DataFrame,
    method: Literal["mean", "median", "bayesian_ridge", "extra_trees"],
    max_allowed_nan_per_column=3,
):

    if method in ["mean", "median"]:
        # Create a SimpleImputer object with the specified strategy
        imputer = SimpleImputer(strategy=method)
    elif method == "bayesian_ridge":
        # Create an IterativeImputer object with BayesianRidge estimator
        imputer = IterativeImputer(estimator=BayesianRidge())
    elif method == "extra_trees":
        # Create an IterativeImputer object with ExtraTreesRegressor estimator
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=10, random_state=0)
        )
    else:
        raise ValueError("Invalid imputation method specified")

    # Loop through each column in the DataFrame
    for column in df.columns:
        # Check if the number of NaNs in the column is less than or equal to max_allowed_nan_per_column
        if df[column].isna().sum() <= max_allowed_nan_per_column:
            # Impute NaN values with the specified method
            df[[column]] = imputer.fit_transform(df[[column]])

    return df


def process_disease_features(df):

    df[DISEASE_FEATURES] = df[DISEASE_FEATURES].applymap(lambda x: 0 if x == 1 else 1)
    df["疾病加總"] = df[DISEASE_FEATURES].sum(axis=1)
    df.drop(columns="有無疾病", inplace=True)

    return df


def select_features(df, indices):

    selected_features = ["Class", "認知退化"] + [SVM_SELECT_FEATURES[i] for i in indices]
    # selected_features = ["Class", "認知退化"] + [RF_SELECT_FEATURES[i] for i in indices]
    # selected_features = ["Class", "認知退化"] + [SVM_RF_SELECT_FEATURES[i] for i in indices]

    return df[selected_features].copy()


def compare_distributions(df, col1, col2):
    """
    Compare the distributions of two columns in a DataFrame using histograms and box plots.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    col1 (str): The name of the first column.
    col2 (str): The name of the second column.
    example: hd_parm.compare_distributions(df, "數字符號", "Tau Mean")
    """
    # Drop rows where both columns have NaN values to make a fair comparison
    data = df[[col1, col2]].dropna(subset=[col1])

    # Plot histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    sns.histplot(data[col1], kde=True, color="blue", label=col1)
    plt.title(f"Histogram of {col1}")
    plt.xlabel(col1)
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(2, 2, 2)
    sns.histplot(data[col2], kde=True, color="orange", label=col2)
    plt.title(f"Histogram of {col2}")
    plt.xlabel(col2)
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(2, 2, 3)
    sns.boxplot(y=data[col1], color="blue")
    plt.title(f"Box Plot of {col1}")
    plt.ylabel(col1)

    plt.subplot(2, 2, 4)
    sns.boxplot(y=data[col2], color="orange")
    plt.title(f"Box Plot of {col2}")
    plt.ylabel(col2)

    plt.tight_layout()
    plt.show()


# Function to normalize the data using MinMaxScaler
def normalize_data(x):
    minmax_scaler = preprocessing.MinMaxScaler()
    return minmax_scaler.fit_transform(x)


# Function to perform PCA on the data
def perform_pca(x, retain):
    if retain == "None":
        return x
    pca = sklearnPCA(n_components=retain)
    return pca.fit_transform(x)

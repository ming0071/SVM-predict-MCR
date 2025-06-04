import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SVMSMOTE

import module.handle_parm as hd_parm
import module.SVM_model as SVM

# Define the classifiers to evaluate
final_result = pd.DataFrame(
    [
        "SVM_Linear",
        "SVM_RBF",
        "SVM_Poly",
    ],
    columns=["Classifier"],
)

# Load the dataset
MCR_excel = "spss_pre_OP_parameter_analysis.xlsx"
df = pd.read_excel("./dataset/" + MCR_excel)

# Data processing
label = [0, 1]  # label 0: good, 1: poor
label_name = "神經功能評估等級mRS(3個月後)"

df = hd_parm.replace_mTICI(df)  # Clean the "手術結果分級mTICI" parameter
df = hd_parm.addclass(df, label, label_name)  # Add class column(= mRS)
df.to_excel("./output/beforeDelete.xlsx", encoding="utf_8_sig")

df = hd_parm.delete(df, label_name, MCR_excel)  # Delete unwanted parameters，剩下38個
df = df.dropna()  # Remove rows with missing data
df.reset_index(drop=True, inplace=True)  # Reset index

# Feature selection
PCA = 0.8  # "None"
ra = 7
y = df["Class"].values
df = df.drop(columns=["Class", "病歷號碼"])
df.to_excel("./output/afterDelete.xlsx", encoding="utf_8_sig")

df = hd_parm.normalize(df)  # Normalize data
df = hd_parm.PCA(df, PCA)  # if PCA set "None" do nothig, else retain set 0.8
X = df

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=ra
)
print("Original :")
print("Y_train  :", sorted(Counter(y_train).items(), key=lambda x: x[1], reverse=True))
print("Y_test   :", sorted(Counter(y_test).items(), key=lambda x: x[1], reverse=True))

# Feature scaling
sm = SVMSMOTE(random_state=20)  # 42 ,20->71
x_train, y_train = sm.fit_resample(x_train, y_train)
print("Feature scaling :")
print("Y_train  :", sorted(Counter(y_train).items(), key=lambda x: x[1], reverse=True))
print("Y_test   :", sorted(Counter(y_test).items(), key=lambda x: x[1], reverse=True))

# Initialize result dictionary
result = {
    "Model": [],
    "PCA": [],
    "Feature_num": [],  # Number of features in the training data
    "Train ACC": [],
    "Test ACC": [],
    "AUC": [],
}

# Compare SVM models with different parameters
result, svcModel = SVM.compare_SVM(result, x_train, x_test, y_train, y_test, label, PCA)

result_1 = pd.DataFrame(result).drop(["Model", "PCA", "Feature_num"], axis=1)
final_result = pd.concat([final_result, result_1], axis=1)

final_result.to_excel("./output/endResult.xlsx", encoding="utf_8_sig")


# Compute average performance metrics
avg = {
    "AVG Train ACC": "{:.2f} ± {:.2f}".format(
        math.floor(final_result["Train ACC"].mean() * 100) / 100.0,
        math.floor(final_result["Train ACC"].std() * 100) / 100.0,
    ),
    "AVG Test ACC": "{:.2f} ± {:.2f}".format(
        math.floor(final_result["Test ACC"].mean() * 100) / 100.0,
        math.floor(final_result["Test ACC"].std() * 100) / 100.0,
    ),
    "AVG AUC": "{:.2f} ± {:.2f}".format(
        math.floor(final_result["AUC"].mean() * 100) / 100.0,
        math.floor(final_result["AUC"].std() * 100) / 100.0,
    ),
}

result["Model"] = ["linear", "rbf", "poly"]
result["Avg Train ACC"] = avg["AVG Train ACC"]
result["Avg Test ACC"] = avg["AVG Test ACC"]
result["Avg AUC"] = avg["AVG AUC"]

# Create a DataFrame with the final results
final_result = pd.DataFrame(result)

# Save the final results to a file
final_result.to_excel("./output/result.xlsx", encoding="utf_8_sig")

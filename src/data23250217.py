import pandas as pd

# Load the dataset
excel_01 = "20250217.xlsx"

df01 = pd.read_excel(f"./dataset/{excel_01}")

merged_df = pd.concat([df01], ignore_index=True)
merged_df.drop(
    columns=[
        "FOGQ總分",
        "GAD-7",
        "第一部分總分",
        "第二部分總分",
        "第二部分總分",
        "第三部分總分",
        "第四部分總分",
        "顫抖/姿勢步態",
        "病歷號",
        "Unnamed: 0",
        "Unnamed: 18",
        "Unnamed: 56",
        "Unnamed: 131",
    ],
    inplace=True,
)
merged_df = merged_df[merged_df["認知退化"] != 9]
merged_df = merged_df[merged_df["分組"].isin(["MCR", "MCI", "2"])]
merged_df["用藥"] = merged_df["用藥"].replace("2,3", 2)
merged_df["用藥"] = merged_df["用藥"].replace("1,7", 1)
merged_df["用藥"] = merged_df["用藥"].replace("1,3", 1)

merged_df.to_excel("./dataset/20250217-result.xlsx", index=False)

import pandas as pd

# Load the dataset
excel_01 = "MMH-MM-MERGED-0614.xlsx"
excel_02 = "handleparm_follow.xlsx"

df01 = pd.read_excel(f"./dataset/{excel_01}")
df02 = pd.read_excel(f"./dataset/{excel_02}")

df02 = df02[["認知退化", "病歷號"]]
merged_df = pd.merge(df01, df02, on="病歷號", how="inner")


merged_df.to_excel("./dataset/MMH-MM-MERGED-0909.xlsx", index=False)

import pandas as pd

# Load the dataset
excel_01 = "MMH-MM-11011-merged-0614.xlsx"
excel_02 = "MMH-MM-11107-merged-0614.xlsx"

df01 = pd.read_excel(f"./dataset/{excel_01}")
df02 = pd.read_excel(f"./dataset/{excel_02}")

merged_df = pd.concat([df01, df02], ignore_index=True)
merged_df.drop(
    columns=["四公尺直線(s)", "taandem stance(s)", "備註2", " 備註"], inplace=True
)
# merged_df["重複"] = merged_df.groupby("受試者姓名").cumcount() + 1

# Remove duplicated rows
original_row_count = merged_df.shape[0]
merged_df.sort_values(by=["受試者姓名", "分組", "神經心理功能測驗日期"], inplace=True)
merged_df = merged_df.drop_duplicates(subset=["受試者姓名"], keep="first")
removed_row_count = original_row_count - merged_df.shape[0]


print(f"移除了 {removed_row_count} 行重複資料")  # 移除了 54 行重複資料
merged_df.reset_index(drop=True, inplace=True)  # 對索引重新編排

merged_df.to_excel("./dataset/MMH-MM-MERGED-0614.xlsx", index=False)

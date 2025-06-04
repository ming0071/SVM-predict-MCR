import pandas as pd


def process_01_data(df, column_mapping=None):
    df = df.iloc[1:].rename(columns=df.iloc[0])
    df.columns = [column_mapping.get(i, df.columns[i]) for i in range(len(df.columns))]
    df = df.drop(df.iloc[60:63].index)

    return df[["受試者姓名", "Tau Mean", "Abeta1-42 Mean", "alpha-synuclein Mean"]]


def process_02_data(df, column_mapping=None):
    df = df.iloc[2:].rename(columns=df.iloc[1])
    df.columns = [column_mapping.get(i, df.columns[i]) for i in range(len(df.columns))]
    df = df.drop(columns=["步態表現評估測驗日期"])
    df = df.drop(columns=["分類字彙流暢", "備註", "備註2"])

    df["教育程度"] = df["教育程度"].replace("日治三年", 1)
    df["握力右"] = df["握力右"].replace("(開刀)無法出力", 28.1)

    return df


def process_03_data(df, column_mapping=None):
    df = df.iloc[1:].rename(columns=df.iloc[0])

    return df[["受試者姓名", "分組"]]


column_01_mapping = {
    6: "Tau Mean",
    9: "Abeta1-42 Mean",
    12: "alpha-synuclein Mean",
}
column_02_mapping = {
    0: "受試者編號",
    1: "病歷號",
    2: "受試者姓名",
    3: "神經心理功能測驗日期",
    4: "步態表現評估測驗日期",
    121: "算術總數2",
    122: "正確數2",
    123: "錯誤數2",
    128: "備註",
    129: "備註2",
}


def main(excel_name, output_name):
    # Load the dataset
    sheet_name01 = "(勿動!!)檢體報告(文君)--MMH-MM-11011"
    sheet_name02 = "測驗分數MMH-MM-11011(于慈)"
    sheet_name03 = "受試者動態-MMH-MM-11011"

    df01 = pd.read_excel(f"./dataset/{excel_name}", sheet_name=sheet_name01)
    df01 = process_01_data(df01, column_01_mapping)

    df02 = pd.read_excel(f"./dataset/{excel_name}", sheet_name=sheet_name02)
    df02 = process_02_data(df02, column_02_mapping)

    df03 = pd.read_excel(f"./dataset/{excel_name}", sheet_name=sheet_name03)
    df03 = process_03_data(df03)

    merged_df = pd.merge(df01, df02, on="受試者姓名", how="inner")
    merged_df = pd.merge(merged_df, df03, on="受試者姓名", how="inner")

    # print(df01.iloc[55:65, 0:4], "\n")
    # print(df02.iloc[0:5, 0:4], "\n")
    # print(df03.iloc[0:5, 0:4], "\n")
    # print(merged_df.head())

    file_path = "./dataset/MMH-MM-11011-sheet-0506.xlsx"
    sheet_names = ["Sheet1", "Sheet2", "Sheet3"]

    with pd.ExcelWriter(file_path) as writer:
        for i, df in enumerate([df01, df02, df03]):
            df.to_excel(writer, sheet_name=sheet_names[i], index=False)

    merged_df.to_excel(f"./dataset/{output_name}", index=False)


if __name__ == "__main__":
    excel_name = "MMH-MM-11011-240614.xlsx"
    output_name = "MMH-MM-11011-merged-0614.xlsx"
    main(excel_name, output_name)

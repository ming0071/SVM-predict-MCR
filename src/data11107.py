import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def process_01_data(df, column_mapping=None):
    df = df.iloc[1:].rename(columns=df.iloc[0])
    df.columns = [column_mapping.get(i, df.columns[i]) for i in range(len(df.columns))]
    df = df.drop(df.iloc[68:].index)
    df = df.dropna(subset=["受試者姓名"])
    df.sort_values(by=["受試者姓名", "收案日期"], inplace=True)
    df.drop_duplicates(subset=["受試者姓名"], keep="first", inplace=True)

    return df[["受試者姓名", "Tau Mean", "Abeta1-42 Mean", "alpha-synuclein Mean"]]


def process_02_data(df, column_mapping=None):
    df = df.iloc[2:].rename(columns=df.iloc[1])
    df.columns = [column_mapping.get(i, df.columns[i]) for i in range(len(df.columns))]
    df = df.dropna(subset=["受試者姓名"])
    df = df.drop(columns=["CDR", "備註"])

    df["視覺疾病"] = df["視覺疾病"].replace(
        {"1(白內障)": 1, "1(視網膜病變已OP)": 1, "1(白內障OP)": 1}
    )
    df["心臟疾病"] = df["心臟疾病"].replace("1(CAD)", 1)
    df["失智症"] = df["失智症"].replace("1(輕度)", 1)
    df["癌症"] = df["癌症"].replace({"1(stomach & bladder)": 1, "1(breast)": 1})

    df.sort_values(by=["受試者姓名", "神經心理功能測驗日期"], inplace=True)
    df.drop_duplicates(subset=["受試者姓名"], keep="first", inplace=True)

    return df


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
    120: "算術總數2",
    121: "正確數2",
    122: "錯誤數2",
    127: "備註",
    128: "備註2",
}


def main(excel_name, output_name):
    # Load the dataset
    sheet_name01 = "(勿動!!)檢體報告(瑋伶)--MMH-MM-11111"
    sheet_name02 = "測驗分數MMH-MM-11111(貞嘉)"

    df01 = pd.read_excel(f"./dataset/{excel_name}", sheet_name=sheet_name01)
    df01 = process_01_data(df01, column_01_mapping)

    df02 = pd.read_excel(f"./dataset/{excel_name}", sheet_name=sheet_name02)
    df02 = process_02_data(df02, column_02_mapping)

    merged_df = pd.merge(df01, df02, on="受試者姓名", how="inner")
    merged_df = merged_df.assign(分組="MCR")

    # print(df01.iloc[55:65, 0:4], "\n")
    # print(df02.iloc[0:5, 0:4], "\n")
    # print(df03.iloc[0:5, 0:4], "\n")
    # print(merged_df.head())

    file_path = "./dataset/MMH-MM-11107-sheet-0506.xlsx"
    sheet_names = ["Sheet1", "Sheet2"]

    with pd.ExcelWriter(file_path) as writer:
        for i, df in enumerate([df01, df02]):
            df.to_excel(writer, sheet_name=sheet_names[i], index=False)

    merged_df.to_excel(f"./dataset/{output_name}", index=False)


if __name__ == "__main__":
    excel_name = "MMH-MM-11107-240614.xlsx"
    output_name = "MMH-MM-11107-merged-0614.xlsx"
    main(excel_name, output_name)

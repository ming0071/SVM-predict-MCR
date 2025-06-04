# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:13:11 2021

@author: user
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# %% 將手術結果分級mTICI 2a 2b 改為 2 & 2.5
# 排除 手術結果爛的(0~2a) 以及 手術後是否有腦出血(僅保留2b、3)
def replace_mTICI(df):

    df["手術結果分級mTICI(0,1,2a,2b,3)\nresult"].replace("2a", 2, inplace=True)
    df["手術結果分級mTICI(0,1,2a,2b,3)\nresult"].replace("2b", 2.5, inplace=True)
    df["手術結果分級mTICI(0,1,2a,2b,3)\nresult"].replace("3", 3, inplace=True)
    filter1 = df["手術結果分級mTICI(0,1,2a,2b,3)\nresult"] == 2.5
    filter2 = df["手術結果分級mTICI(0,1,2a,2b,3)\nresult"] == 3
    # filter3 = df["是否產生有症狀腦出血(0:No, 1:Yes)"] == 0
    # df.where((filter1 | filter2) & filter3, inplace = True)
    df = df[(filter1) | (filter2)]
    df.reset_index(drop=True, inplace=True)  # 對索引重新編排
    # %%
    # original = ['出院使用之抗凝血藥物(0:No, 1:Rivaroxaban, 2:Apixaban, 3:Dabigatran, 4:Edoxaban, 5:Warfarin)',
    #             '出院有無使用降血脂藥物(0:No, 1:yes)',
    #             '有無使用降血醣藥物(0:No, 1:Yes)',
    #             '有無使用高血壓藥物(0:No, 1:Yes)',
    #             '是否有使用高血壓藥物-ARB類(0:No, 1:Yes)',
    #             '是否有使用高血壓藥物-A-block類(0:No, 1:yes)',
    #             '是否有使用高血壓藥物-B-block類(0:No, 1:Yes)',
    #             '是否有使用高血壓藥物-Loop類(0:No, 1:Yes)',
    #             '是否有使用高血壓藥物-Actone類(0:No, 1:Yes)',
    #             '出院使用之抗血小板藥物(0:No, 1:Aspirin, 2: Plavix, 3:A+P, 4:A+Cilostazol)',
    #             '是否有使用高血壓藥物-CCB類(0:No, 1:Yes)']

    # for i in original:
    #     df[i].replace(np.nan,0,inplace = True)

    # df['手術中電腦斷層是否有出血(0:No, 1: Yes, 空值:未執行)'].replace(np.nan,2,inplace = True)
    # df['發作至施打t-PA時間(分)'].replace(np.nan,500,inplace = True)
    # df['到院至施打t-PA時間(分)'].replace(np.nan,500,inplace = True)

    return df


# %%  得到相同Test 數量
pca = 0.8


def balance(df, pra_name, pca):
    # 0 類有 21人
    ra = 42
    df0 = df.where(df["Class"] == 0).dropna()
    y_test0, y_train0 = train_test_split(
        df0["Class"], train_size=10, random_state=ra
    )  # 10 為保留10個做TEST

    # 1 類有 61人
    df1 = df.where(df["Class"] == 1).dropna()
    y_test1, y_train1 = train_test_split(
        df1["Class"], train_size=10, random_state=ra
    )  # 10 為保留10個做TEST

    y_train = pd.concat([y_train0, y_train1])
    y_train = y_train.sample(frac=1, random_state=ra).reset_index(drop=True).values

    y_test = pd.concat([y_test0, y_test1])
    y_test = y_test.sample(frac=1, random_state=ra).reset_index(drop=True).values

    x_test0, x_train0 = train_test_split(
        df0, train_size=10, random_state=ra
    )  # 10 為保留10個做TEST
    x_test1, x_train1 = train_test_split(
        df1, train_size=10, random_state=ra
    )  # 10 為保留10個做TEST

    if pca == 100:
        # 原始
        # 　train  0 : 11人  1:  51 人
        #  test   0 : 10人  1:  10 人
        #  train 0 放大 7 倍
        df_train = pd.concat([x_train0, x_train1])
        df_train = df_train.sample(frac=1, random_state=ra).reset_index(drop=True)

        df_test = pd.concat([x_test0, x_test1])
        df_test = df_test.sample(frac=1, random_state=ra).reset_index(drop=True)
        return df_train, df_test
    else:
        x = pd.concat([x_train0, x_train1, x_test0, x_test1]).reset_index(drop=True)
        if pra_name == "None":
            x = x.drop(columns=["Class", "病歷號碼"])
        else:
            x = x.drop(columns=["Class", "病歷號碼"] + pra_name)
        x = normalize(x)  # 正規化
        x = PCA(x, pca)  # PCA 0.8

        # 將X分成 train & test
        x = pd.DataFrame(x)
        x_train = x[: len(x_train0) + len(x_train1)]
        x_test = x[-(len(x_test0) + len(x_test1)):]
        x_train = (
            x_train.sample(frac=1, random_state=ra).reset_index(drop=True).to_numpy()
        )
        x_test = (
            x_test.sample(frac=1, random_state=ra).reset_index(drop=True).to_numpy()
        )

        return y_train, y_test, x_train, x_test


# %% 建立 label:0 為好，1 為差。mTICI 術後恢復狀況
# mRS  :0~2   -> 0, 3~6   -> 1
# mTICI:0~2.5 -> 1, 2.5~3 -> 0
def addclass(df, label, col):
    b = []
    for x in range(df[col].size):
        if np.isnan(df[col][x]):
            b.append(np.nan)
        else:
            if df[col][x] <= 2.5:  # TODO:自己有改 原2.5，我2
                b.append(label[0])  # 0 皆為好的
            #     elif 0.5 < df[col][x] < 4:
            #         b.append(1)
            else:
                b.append(label[1])
    df["Class"] = b
    return df


# %% 刪除不要的資料
def delete(df, label, filename):
    #  移除確定不要的參數，藍色的
    df = df.drop(
        columns=[
            "Name",
            "發作時間",
            "到院時間",
            "t-PA時間",
            "不打藥電腦斷層執行時間",
            "打藥電腦斷層執行時間",
            "取栓手術執行開始時間",
            "血栓打通時間或結束手術時間",
            "神經學症狀評估NIHSS(第一次出ICU)",
            "神經學症狀評估NIHSS(一般病房)",
            "神經學症狀評估NIHSS(出院)",
            "心室輸出率(%)",
            "發作至施打t-PA時間(分)",
            "到院至施打t-PA時間(分)",
            "有無使用抗癲癇藥物(0:No, 1:yes)",
            "手術中電腦斷層是否有出血(0:No, 1: Yes, 空值:未執行)",
            "神經學症狀評估NIHSS(tPAorEVT後)",
            "出院使用之抗凝血藥物(0:No, 1:Rivaroxaban, 2:Apixaban, 3:Dabigatran, 4:Edoxaban, 5:Warfarin)",
            "出院有無使用降血脂藥物(0:No, 1:yes)",
            "有無使用降血醣藥物(0:No, 1:Yes)",
            "有無使用高血壓藥物(0:No, 1:Yes)",
            "是否有使用高血壓藥物-ARB類(0:No, 1:Yes)",
            "是否有使用高血壓藥物-A-block類(0:No, 1:yes)",
            "是否有使用高血壓藥物-B-block類(0:No, 1:Yes)",
            "是否有使用高血壓藥物-Loop類(0:No, 1:Yes)",
            "是否有使用高血壓藥物-Actone類(0:No, 1:Yes)",
            "是否有使用高血壓藥物-CCB類(0:No, 1:Yes)",
            "出院使用之抗血小板藥物(0:No, 1:Aspirin, 2: Plavix, 3:A+P, 4:A+Cilostazol)",
            "神經功能評估等級mRS(出院當下)",
            '電腦斷層缺血分數ASPECTS_score',
            '電腦斷層側枝循環良好程度分級(0:poor, 1:intermediate, 2:good)'
        ]
    )
    # 原版
    # df = df.drop(columns=['Name','發作時間','到院時間','t-PA時間',
    #                       '不打藥電腦斷層執行時間','打藥電腦斷層執行時間',
    #                       '取栓手術執行開始時間','血栓打通時間或結束手術時間',
    #                       '神經學症狀評估NIHSS(第一次出ICU)',
    #                       '神經學症狀評估NIHSS(一般病房)',
    #                       '神經學症狀評估NIHSS(出院)','心室輸出率(%)',
    #                       '神經學症狀評估NIHSS(tPAorEVT後)'
    #                       ]) #  移除確定不要的參數

    # 將空白資料 & 排除資料 變成 NAN
    Indicator_results = [
        "CT影像(完整:1, 不全:0)",
        "CTA影像(完整:1, 不全:0)",
        "分析(納入:1, 排除:0)",
        "mRS1分析(納入:1, 排除:0)",
        "mRS3分析(納入:1, 排除:0)",
        "mRS6分析(納入:1, 排除:0)",
    ]
    # 把 Indicator_results 空的變為 NaN，label 空的和0變為 NaN
    for i in Indicator_results + [label]:
        if i == label:
            df.loc[(df[i] == ""), i] = np.nan
            break
        df.loc[(df[i] == "") | (df[i] == 0), i] = np.nan

    # 結合 label 用 mRSX分析 決定要不要排除
    if label == "神經功能評估等級mRS(1個月後)":
        label1 = [label, "mRS1分析(納入:1, 排除:0)"]
    elif label == "神經功能評估等級mRS(3個月後)":
        label1 = [label, "mRS3分析(納入:1, 排除:0)"]
    elif label == "神經功能評估等級mRS(6個月後)":
        label1 = [label, "mRS6分析(納入:1, 排除:0)"]

    df.dropna(
        subset=label1, inplace=True
    )  # shape (87,73) 用label1決定要不要排除 df 有缺失的 row
    df.reset_index(drop=True, inplace=True)  # 對索引重新編排

    # 以 mRS 作為預測目標  移除相關參數
    # all_label = ["神經功能評估等級mRS(出院當下)",
    #              "神經功能評估等級mRS(1個月後)",
    #              "神經功能評估等級mRS(3個月後)",
    #              "神經功能評估等級mRS(6個月後)"
    #             ] + Indicator_results
    all_label = [
        "神經功能評估等級mRS(1個月後)",
        "神經功能評估等級mRS(3個月後)",
        "神經功能評估等級mRS(6個月後)",
    ] + Indicator_results

    if label in all_label:
        # all_label.remove(label)
        df = df.drop(columns=all_label)

    # 手術前狀況預測mRS
    if filename == "spss_pre_OP_parameter_analysis.xlsx":  # TODO:自己有改
        #  移除確定不要的參數
        df = df.drop(
            columns=[
                "住院中使用之抗血小板藥物(0:No, 1:Aspirin, 2:Plavix, 3:A+P)",
                "住院中使用抗凝血藥物(0:No, 1:Rivaroxaban, 2:Apixaban, 3:Dabigatran, 4:Edoxaban, 5:Warfarin)",
                "住院中有無使用降血脂藥物(0:No, 1:Yes)",
                "手術總時間(分)",
                "症狀至打通或手術結束時間(分)",
                "入院至打通或手術結束時間(分)",
                "手術結果分級mTICI(0,1,2a,2b,3)\nresult",
                "手術抽吸次數",
                "手術拉栓次數",
                "是否產生有症狀腦出血(0:No, 1:Yes)",
                "是否執行開顱手術(0:No, 1:yes)",
            ]
        )

    # 手術結果當預測目標  移除參數
    if label == "手術結果分級mTICI(0,1,2a,2b,3)\nresult":
        remove_param = [
            "手術結果分級mTICI(0,1,2a,2b,3)\nresult",
            "住院中使用之抗血小板藥物(0:No, 1:Aspirin, 2:Plavix, 3:A+P)",
            "住院中使用抗凝血藥物(0:No, 1:Rivaroxaban, 2:Apixaban, 3:Dabigatran, 4:Edoxaban, 5:Warfarin)",
            "住院中有無使用降血脂藥物(0:No, 1:Yes)",
            "出院使用之抗血小板藥物(0:No, 1:Aspirin, 2: Plavix, 3:A+P, 4:A+Cilostazol)",
            "出院使用之抗凝血藥物(0:No, 1:Rivaroxaban, 2:Apixaban, 3:Dabigatran, 4:Edoxaban, 5:Warfarin)",
            "出院有無使用降血脂藥物(0:No, 1:yes)",
            "手術中電腦斷層是否有出血(0:No, 1: Yes, 空值:未執行)",
            "是否產生有症狀腦出血(0:No, 1:Yes)",
            "是否執行開顱手術(0:No, 1:yes)",
            "神經功能評估等級mRS(出院當下)",
            "神經功能評估等級mRS(1個月後)",
            "神經功能評估等級mRS(3個月後)",
            "神經功能評估等級mRS(6個月後)",
        ]
        # '手術總時間(分)','手術抽吸次數','手術拉栓次數']
        if label in remove_param:
            df = df.drop(columns=remove_param)

    return df


# %% 只保留 DT_important_feature


def DT_important_feature(df, DT_drop_nonimportant):
    df = df.drop(labels=df.columns[DT_drop_nonimportant], axis=1)
    return df


# %% 目標預測手術結果  保留醫師認可的14項參數


def doc_retain(df, three):

    retain_important = [
        "年紀(歲)",
        "是否施打t-PA(0:No, 1:Yes)",
        "是否有顱外內頸動脈中度以上狹窄(0:No, 1:yes)",
        "是否有顱內血管狹窄(0:No, 1:Yes)",
        "心率表現(0:正常SR, 1:心律不整Af)",
        "血脂肪檢驗值LDL",
        "中風機轉分類TOAST(1:LAA, 2:Small vessel, 3:Cardioembolism, 4:Others, 5:Undertermined)",
        "意識狀況分數GCS總分",
        "血栓位置(0:ICA, 1:M1, 2:M2, 3:VA, 4:BA)",
        "症狀至開始手術時間(分)",
        "手術總時間(分)",
        "手術抽吸次數",
        "手術拉栓次數",
        "Class",
    ]
    df = df[retain_important]

    if three == "second":
        df = df.drop(columns=["手術總時間(分)", "手術抽吸次數", "手術拉栓次數"])

    return df


# %% 目標預測 mRs  保留醫師認可的19項參數


def doc_retain_mRS(df):

    retain_important = pd.read_excel(
        "D:/institute/MacKay Memorial Hospital"
        "/MMH_Proposal/2021 CVE/python/CVE/excel"
        "/mRS_reduce_pra.xlsx",
        engine="openpyxl",
        header=None,
    )

    retain_important = retain_important[0].tolist()
    retain_important = ["病歷號碼"] + retain_important + df.columns[-7:].tolist()
    df = df[retain_important]

    return df


# %% 正規化
def normalize(df):
    from sklearn import preprocessing

    minmax = preprocessing.MinMaxScaler()
    # for i in df.columns: # 排除該參數做正規化
    #     if i == '手術結果分級mTICI(0,1,2a,2b,3)\nresult':
    #         continue
    #     df[[i]] = minmax.fit_transform(x[[i]])

    df = minmax.fit_transform(df)
    return df


# %% PCA
def PCA(df, retain):
    if retain == "None":
        return df
    else:
        from sklearn.decomposition import PCA

        #  0.8
        pca = PCA(n_components=retain)
        df = pca.fit_transform(df)
        return df


# %% split train & test


def split_train_test(x, y, percentage, seed):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=percentage, random_state=seed
    )

    print("Training  data shape : ", x_train.shape)
    print("Testing   data shape : ", x_test.shape, "\n")

    unique, counts = np.unique(y_train, return_counts=True)
    print("Y_Train label：", dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Y_Test  label：", dict(zip(unique, counts)))
    return x_train, x_test, y_train, y_test

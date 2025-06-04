"""
這段程式碼是一個 SVM 分類器的實現示例，用於預測社交網絡用戶是否購買了特定商品。程式的主要步驟包括：

1. 載入資料集：從 CSV 文件中讀取數據集。
2. 選取特徵變數和目標變數：從數據集中選取特徵和目標列。
3. 將資料集切割成訓練集和測試集：使用 train_test_split 函數將數據集切割成訓練集和測試集。
4. 特徵縮放：對特徵變數進行標準化處理。
5. 建立 SVM 分類器：使用支持向量機 (SVM) 構建分類器，這裡使用了 RBF 核函數。
6. 預測測試集結果：使用訓練好的模型對測試集進行預測。
7. 評估模型效能：計算模型的準確率並打印出來。
8. 可視化結果：將訓練集和測試集的分類情況可視化展示。

# Reference
https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%2013%20SVM.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# 載入資料集
ADs_excelname = "Social_Network_Ads.csv"
dataset = pd.read_csv("./dataset/" + ADs_excelname)

# 選取特徵變數和目標變數
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# 將資料集切割成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 特徵縮放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# 建立 SVM 分類器 linear accuracy:0.88, rbf accuracy:0.93
classifier = SVC(kernel="rbf", random_state=0)
# classifier = SVC(kernel = 'poly',degree=3,  random_state = 0)
classifier.fit(X_train, y_train)

# 預測測試集結果
y_pred = classifier.predict(X_test)

# 評估模型效能
cm = confusion_matrix(y_test, y_pred)

# visualize train set results
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(("red", "green")),
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        color=ListedColormap(("red", "green"))(i),
        label=j,
    )
plt.title("SVM (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

# visualize test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(("red", "green")),
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        color=ListedColormap(("red", "green"))(i),
        label=j,
    )
plt.title("SVM (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
# print("#error@test: ", cm[0, 1] + cm[0, 1])
print("Accuracy:", accuracy)

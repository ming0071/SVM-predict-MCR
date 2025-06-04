from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 載入資料集
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(iris.target)

# 將資料集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練SVM模型
svm_model = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
svm_model.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = svm_model.predict(X_test)

# 評估模型準確率
accuracy = accuracy_score(y_test, y_pred)
print("準確率:", accuracy)

# 列印分類報告
report = classification_report(y_test, y_pred)
print("分類報告:\n", report)

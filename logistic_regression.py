import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer = datasets.load_breast_cancer()
print(cancer.keys())

df = pd.DataFrame(cancer.data, columns= cancer.feature_names)
# df["label"] = cancer.target
print(df.head())

# print(df)
# print(set(cancer.target))

X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=1234)

# print(X_train)
# print(y_train)

clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
# print(predict)

score_train = clf.score(X_train, y_train)
print('score(train): ' + str(score_train))

score_test = clf.score(X_test, y_test)
print('score(test): ' + str(score_test))

from sklearn.metrics import roc_curve, auc

y_score = clf.predict_proba(X_test)[:, 1] # 検証データがクラス1に属する確率
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)

plt.plot(fpr, tpr, label='roc curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], linestyle='--', label='random')
plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', label='ideal')
plt.legend()
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()

from sklearn.svm import SVC
from sklearn import preprocessing
from extract_drugs import getFeatureVectorAndLabels
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from sklearn.metrics import f1_score, precision_score

FOLDS = 10

np.set_printoptions(threshold=np.inf)
data_dir = "../data/training_data"
files, drugs, features, X, y = getFeatureVectorAndLabels(data_dir)
labelEncoder = preprocessing.LabelEncoder()
y_encoded = labelEncoder.fit_transform(y)
skf = StratifiedKFold(n_splits=FOLDS)
label1 = 0
label2 = 0
label3 = 0
for train, test in skf.split(X, y_encoded):
    SVMclassifier = SVC(kernel = 'rbf',  class_weight='balanced', C = 1000.0, gamma = 'auto')
    SVMclassifier.fit(X[train], y_encoded[train])
    y_pred = SVMclassifier.predict(X[test])
    score = precision_score(y_encoded[test], y_pred, average=None)
    label1 += score[0]
    label2 += score[1]
    label3 += score[2]
    print(score)

print("Average for class 1: " + str(label1.item() / FOLDS))
print("Average for class 2: " + str(label2.item() / FOLDS))
print("Average for class 3: " + str(label3.item() / FOLDS))

from sklearn.svm import SVC
from sklearn import preprocessing
from extract_drugs import getFeatureVectorAndLabels
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from ruleBasedAfterClass import getRuleBasedClassification
from collections import defaultdict

FOLDS = 8
featureNames = ['sections', 'containsFutureWord', 'prevSentContainsFutureWord',\
                'current_tense', 'prev_tense', 'temporalType',\
                'polarity', 'position', 'modality', 'proximity', 'futureCount']

np.set_printoptions(threshold=np.inf)
data_dir = "../data/training_data"
files, drugs, features, X, y = getFeatureVectorAndLabels(data_dir)
labelEncoder = preprocessing.LabelEncoder()
y_encoded = labelEncoder.fit_transform(y)

skf = StratifiedKFold(n_splits=10)
label1 = 0
label2 = 0
label3 = 0
average_accuracy_score = 0
for train, test in skf.split(X, y):
    SVMclassifier = SVC(kernel = 'rbf',  class_weight='balanced', C = 100.0, gamma = 'auto')
    SVMclassifier.fit(X[train], y[train])
    y_SVM_pred = SVMclassifier.predict(X[test])
    features_test = defaultdict(list)
    for feature in featureNames:
        featureList = features[feature]
        features_test[feature] = ((np.array(featureList))[test]).tolist()
    y_rules_pred = getRuleBasedClassification(features_test)
    y_rules_pred_encoded = labelEncoder.fit_transform(y_rules_pred)
    y_pred = []
    for i in range(len(y_SVM_pred)):
        if y_rules_pred[i] == 'after':
            y_pred.append(y_rules_pred_encoded[i])
        else:
            y_pred.append(y_SVM_pred[i])

    score = f1_score(y[test], y_pred, average=None)
    average_accuracy_score += accuracy_score(y[test], y_pred)
    label1 += score[0]
    label2 += score[1]
    label3 += score[2]
    print(score)

print("Average for class 1: " + str(label1.item() / 10))
print("Average for class 2: " + str(label2.item() / 10))
print("Average for class 3: " + str(label3.item() / 10))
print("Average accuracy is: " + str(average_accuracy_score.item() / 10))

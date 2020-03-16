from sklearn.svm import SVC
from sklearn import preprocessing
from joblib import dump,load
import numpy as np
import os
import csv
from stanfordnlp.server import CoreNLPClient
from sklearn.metrics import confusion_matrix, classification_report
from extract_drugs import getFeatureVectorAndLabels
from ruleBasedAfterClass import getRuleBasedClassification
from collections import defaultdict

featureNames = ['sections', 'containsFutureWord', 'prevSentContainsFutureWord',\
                'current_tense', 'prev_tense', 'temporalType',\
                'polarity', 'position', 'modality', 'proximity', 'futureCount']

data_dir = "../data/test_data"

np.set_printoptions(threshold=np.inf)

if os.path.isfile('svmModelC.joblib'):
    SVMclassifier = load('svmModelC.joblib')
else:
    allDrugEvents, features_train, X_train, y_train = getFeatureVectorAndLabels(data_dir)
    labelEncoder = preprocessing.LabelEncoder()
    y_train_encoded = labelEncoder.fit_transform(y_train)
    SVMclassifier = SVC(kernel = 'rbf',  class_weight='balanced', C = 1000.0, gamma = 'auto')
    SVMclassifier.fit(X_train, y_train_encoded)
    dump(SVMclassifier, 'svmModelC.joblib')

sentences, drugs, features, X, y = getFeatureVectorAndLabels(data_dir)

labelEncoder = preprocessing.LabelEncoder()
y_encoded = labelEncoder.fit_transform(y)

y_SVM_pred = SVMclassifier.predict(X)
y_SVM_pred = labelEncoder.inverse_transform(y_SVM_pred)

y_rules_pred = getRuleBasedClassification(features)
y_pred = []
for i in range(len(y_SVM_pred)):
    if y_rules_pred[i] == 'after':
        y_pred.append(y_rules_pred[i])
    else:
        y_pred.append(y_SVM_pred[i])

y_test = labelEncoder.inverse_transform(y_encoded)
with open('drugClassification.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    for i in range(len(y_pred)):
        filewriter.writerow([drugs[i], y_pred[i], y_test[i]])
    filewriter.writerow(["", "", ""])

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

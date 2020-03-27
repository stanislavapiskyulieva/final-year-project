from sklearn.svm import SVC
from sklearn import preprocessing
from extract_drugs import getFeatureVectorAndLabels
from joblib import dump,load
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import csv
from collections import defaultdict
from ruleBasedAfterClass import getRuleBasedClassification

featureNames = ['sections', 'containsFutureWord', 'tense', 'temporalType',\
                'polarity', 'position', 'modality', 'proximity', 'futureCount']

np.set_printoptions(threshold=np.inf)
data_dir = "../data/demo"
files, drugs, features, X, y = getFeatureVectorAndLabels(data_dir)
svmModel = load('svmModelC.joblib')
y_SVM = svmModel.predict(X)
labelEncoder = preprocessing.LabelEncoder()
y = labelEncoder.fit_transform(y)

y_SVM = labelEncoder.inverse_transform(y_SVM)

y_rules_pred = getRuleBasedClassification(features)

y_pred = []
for i in range(len(y_SVM)):
    if y_rules_pred[i] == 'after':
        y_pred.append(y_rules_pred[i])
    else:
        y_pred.append(y_SVM[i])

y = labelEncoder.inverse_transform(y)
with open('demo.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['Drug', 'Predicted Label', 'Correct Label'])
    for i in range(len(y_pred)):
        filewriter.writerow([drugs[i], y_pred[i], y[i]])

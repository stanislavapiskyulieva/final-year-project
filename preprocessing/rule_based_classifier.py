import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
from extract_drugs import getFeatureVectorAndLabels
from collections import defaultdict
import sys

featureNames = ['sections', 'containsFutureWord', 'prevSentContainsFutureWord',\
                'current_tense', 'prev_tense', 'temporalType',\
                'polarity', 'position', 'modality', 'proximity', 'futureCount']

data_dir = "../data/test_data"

np.set_printoptions(threshold=np.inf)

files, drugs, features, X, y = getFeatureVectorAndLabels(data_dir)

sectionsFeatureVector = list(features['sections'])
currTenseFeatureVector = list(features['current_tense'])
containsFutureWordsVector = list(features['containsFutureWord'])

predictedLabels = []

for i in range(len(X)):
    if sectionsFeatureVector[i] == "1" and currTenseFeatureVector[i] == "VBD":
        predictedLabels.append("before")
    elif sectionsFeatureVector[i] == "2" and containsFutureWordsVector[i] != "n/a" or currTenseFeatureVector[i] == "MD":
        predictedLabels.append("after")
    else:
        predictedLabels.append("during")

print(confusion_matrix(y,predictedLabels))
print(classification_report(y,predictedLabels))

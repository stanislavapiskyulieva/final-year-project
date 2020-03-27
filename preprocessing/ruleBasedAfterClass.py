import numpy as np
import os
from extract_drugs import getFeatureVectorAndLabels
from collections import defaultdict
import sys

FOLDS = 5

featureNames = ['sections', 'containsFutureWord', 'prevSentContainsFutureWord',\
                'current_tense', 'prev_tense', 'temporalType',\
                'polarity', 'position', 'modality', 'proximity', 'futureCount']

def getRuleBasedClassification(features):
    allPredictedLabels = []

    sectionsFeatureVector = list(features['sections'])
    containsFutureWordsVector = list(features['containsFutureWord'])
    prevSentContainsFutureWordVector = list(features['prevSentContainsFutureWord'])
    proximityToFutureWordFeatureVector = list(features['proximity'])
    futureWordsCountFeatureVector = list(features['futureCount'])
    currTenseFeatureVector = list(features['current_tense'])
    prevTenseFeatureVector = list(features['prev_tense'])
    drugEventModalityFeatureVector = list(features['modality'])
    positionInTextFeatureVector = list(features['position'])
    for index in range(len(sectionsFeatureVector)):
        if sectionsFeatureVector[index] != '2':
            allPredictedLabels.append("during")

        elif proximityToFutureWordFeatureVector[index] == 0:
            allPredictedLabels.append("after")

        elif (currTenseFeatureVector[index] == 'VBD' or currTenseFeatureVector[index] == 'VBN') \
        and proximityToFutureWordFeatureVector[index] != 0 \
        and (containsFutureWordsVector[index] == 'n/a' and prevSentContainsFutureWordVector[index] == 'n/a')\
        and ((positionInTextFeatureVector[index] < 0.7  \
        and futureWordsCountFeatureVector[index] < 3 \
        and proximityToFutureWordFeatureVector[index] > 5)
        or (futureWordsCountFeatureVector[index] == 1 and proximityToFutureWordFeatureVector[index] > 5)):
            allPredictedLabels.append("during")

        elif sectionsFeatureVector[index] == '2' \
        and ((containsFutureWordsVector[index] != "n/a" and ((positionInTextFeatureVector[index] > 0.5 and proximityToFutureWordFeatureVector[index] <= 7) or proximityToFutureWordFeatureVector[index] < 4 or futureWordsCountFeatureVector[index] > 1)) \
        or currTenseFeatureVector[index] == 'MD' and (prevTenseFeatureVector[index] == 'MD' or positionInTextFeatureVector[index] > 0.8  or containsFutureWordsVector[index] != "n/a") \
        or (positionInTextFeatureVector[index] > 0.7 and (drugEventModalityFeatureVector[index] == 'PROPOSED' or drugEventModalityFeatureVector[index] == 'CONDITIONAL'))):
            allPredictedLabels.append("after")
        else:
            allPredictedLabels.append("during")

    return allPredictedLabels

from sklearn.svm import SVC
from sklearn import preprocessing
from joblib import dump,load
from nltk.tokenize import PunktSentenceTokenizer
import numpy as np
import os
from stanfordnlp.server import CoreNLPClient
from extract_labels import getLabels
from sklearn.metrics import confusion_matrix, classification_report
from extract_drugs import getPositionInTextFeatureVector, getAllDrugsFromCLAMP, getSectionFeature, getContainsFutureWordsFeature, getTenseFeatureVector, getDrugEvents, flattenValues

np.set_printoptions(threshold=np.inf)
data_dir = "../data/training_data"
allCorrectLabels = []
allPredictedLabels = []
training_text = ""
corpus = []
dischargeWords = ['discharged on', 'discharged home on', 'discharged to home on']
coreNLPClient = CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], timeout=100000, memory='8G')
for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            f = open(os.path.join(data_dir, file), 'r')
            raw = f.read()
            training_text += raw
            corpus.append(raw.lower())
punkt_sent_tokenizer = PunktSentenceTokenizer(training_text)
for file in os.listdir(data_dir):
    if not file.endswith('.txt') or not os.path.isfile(os.path.join(data_dir, os.path.splitext(file)[0])):
        continue
    f = open(os.path.join(data_dir, file), 'r')
    raw = f.read()
    CLAMPdrugs = getAllDrugsFromCLAMP(file, data_dir, raw)
    drugEvents, drugEventsStartIndices, drugEventPolarityFeatureVector, drugEventModalityFeatureVector = getDrugEvents(file, data_dir, CLAMPdrugs)

    correctLabels = getLabels(file, drugEvents, data_dir)
    allCorrectLabels.extend(correctLabels)
    sectionsFeatureVector = getSectionFeature(file, data_dir, drugEventsStartIndices)
    containsFutureWordsVector, proximityToFutureWordFeatureVector, futureWordsCountFeatureVector = getContainsFutureWordsFeature(raw, drugEvents)
    tenseFeatureVector = getTenseFeatureVector(file, coreNLPClient, drugEvents, raw)
    positionInTextFeatureVector = getPositionInTextFeatureVector(raw, drugEvents)

    drugEventsSet = drugEvents.keys()
    drugEventsList, drugEventIndicesFlatMap = flattenValues(drugEvents)
    drugEventIndicesFlatMap.sort()

    nltkSentences = punkt_sent_tokenizer.tokenize(raw)
    sentenceSpans = list(punkt_sent_tokenizer.span_tokenize(raw))

    print(drugEventsSet)
    for i in range(len(nltkSentences)):
        if not any(drugEvent in nltkSentences[i] for drugEvent in drugEventsSet):
            continue
        for index, startInd in enumerate(drugEventIndicesFlatMap):
            if startInd >= sentenceSpans[i][0] and startInd <= sentenceSpans[i][1] + 1:
                if correctLabels[index] == 'after':
                    print(nltkSentences[i])

                if sectionsFeatureVector[index] != '2':
                    allPredictedLabels.append("during")

                elif (tenseFeatureVector[index] == 'VBD' or tenseFeatureVector[index] == 'VBN' or tenseFeatureVector[index] == 'VBP' and futureWordsCountFeatureVector[index] == 1) \
                and proximityToFutureWordFeatureVector[index] != 0 \
                and ((positionInTextFeatureVector[index] < 0.7  \
                and futureWordsCountFeatureVector[index] < 3 \
                and proximityToFutureWordFeatureVector[index] > 3) \
                or (positionInTextFeatureVector[index] > 0.7 and futureWordsCountFeatureVector[index] == 1 and proximityToFutureWordFeatureVector[index] < 5)):
                    allPredictedLabels.append("during")

                elif (containsFutureWordsVector[index] != "n/a" and (positionInTextFeatureVector[index] > 0.8 or proximityToFutureWordFeatureVector[index] < 4)) \
                or tenseFeatureVector[index] == 'MD' and (positionInTextFeatureVector[index] > 0.8  or containsFutureWordsVector[index] != "n/a") \
                or (positionInTextFeatureVector[index] > 0.7 and (drugEventModalityFeatureVector[index] == 'PROPOSED' or drugEventModalityFeatureVector[index] == 'CONDITIONAL')) \
                or proximityToFutureWordFeatureVector[index] == 0:
                    allPredictedLabels.append("after")
                else:
                    allPredictedLabels.append("during")
correctPredictions = 0
afterClass = 0
for i, label in enumerate(allCorrectLabels):
    if label == 'after':
        afterClass += 1
    if label == 'after' and label == allPredictedLabels[i]:
        correctPredictions += 1
print("Accuracy is " + str(correctPredictions / afterClass * 100))
labels = ['after', 'other']
print(confusion_matrix(allCorrectLabels, allPredictedLabels))
print(classification_report(allCorrectLabels,allPredictedLabels))

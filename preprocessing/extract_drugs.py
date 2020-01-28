import os
import xml.etree.ElementTree as ET
import nltk
import csv
import re
from nltk.tokenize import sent_tokenize, WhitespaceTokenizer, word_tokenize, regexp_tokenize, RegexpTokenizer, PunktSentenceTokenizer
from string import punctuation
from xml.dom import minidom
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from numpy import sum
from extract_labels import getLabels
from sklearn.metrics import confusion_matrix, classification_report
from stanfordnlp.server import CoreNLPClient
import requests
import numpy as np
from matplotlib import pyplot

patterns = r'''(?x)
         \w+
        '''

data_dir = "../data/training_data"
futureWords = ['further', 'future', 'home', 'follow', 'follow-up']
tensePOStags = ['MD', 'VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB']
FUTURE_WORDS_RULE = True
FEATURES_SIZE = 3
nltkTokenizer = RegexpTokenizer(patterns)
tfidf_vectorizer=TfidfVectorizer(smooth_idf=True,use_idf=True)
training_text = ""
corpus = []
for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            f = open(os.path.join(data_dir, file), 'r')
            raw = f.read()
            training_text += raw
            corpus.append(raw)
punkt_sent_tokenizer = PunktSentenceTokenizer(training_text)
tfidf_vectorizer.fit(corpus)

def getFirstVP(constituencyTree):
    queue = [constituencyTree]
    while len(queue) != 0:
        currentNode = queue.pop(0)
        if currentNode.value == "VP":
            return currentNode
        for child in currentNode.child:
            queue.append(child)

def appendTense(VPnode, drugsInSentence, featureVector):
    for child in VPnode.child:
        if child.value in tensePOStags:
            while drugsInSentence > 0:
                featureVector.append(child.value)
                drugsInSentence-=1
            return drugsInSentence
    return drugsInSentence

def flattenValues(drugEvents):
    drugEventIndicesFlatMap = []
    for drug in drugEvents:
        for i in drugEvents[drug]:
            drugEventIndicesFlatMap.append(i)
    return drugEventIndicesFlatMap

def flattenKeys(drugEvents):
    drugEventsFlatMap = []
    for drug in drugEvents:
        for i in drugEvents[drug]:
            drugEventsFlatMap.append(drug)
    return drugEventsFlatMap

def getTemporalCluesFeatureVectors(fileName, drugEvents, raw, data_dir):
    temporalTypeFeatureVector = []
    data_dir_features = data_dir + "/section_ids/"
    xmldoc = minidom.parse(data_dir_features + os.path.splitext(fileName)[0]+'.xmi')
    entities = xmldoc.getElementsByTagName("typesystem:ClampNameEntityUIMA")
    temporalExpressions = [temp for temp in entities if temp.getAttribute('semanticTag') == 'temporal']

    nltkSentences = punkt_sent_tokenizer.tokenize(raw)
    sentenceSpans = list(punkt_sent_tokenizer.span_tokenize(raw))
    for i in range(len(nltkSentences)):
        if not any(drugEvent in nltkSentences[i] for drugEvent in drugEvents.keys()):
            continue

        drugEventsSet = drugEvents.keys()
        drugEventIndicesFlatMap = flattenValues(drugEvents)
        drugsInSentence = sum(list(map(lambda drugEventIndex : drugEventIndex >= sentenceSpans[i][0] + 1 and drugEventIndex <= sentenceSpans[i][1] + 1, drugEventIndicesFlatMap)))

        tempType = ""
        for temp in temporalExpressions:
            if int(temp.getAttribute('begin')) >= sentenceSpans[i][0] and int(temp.getAttribute('begin')) <= sentenceSpans[i][1]:
                tempList = re.split("\|\|", temp.getAttribute('attr1'))
                tempType = tempList[0]
                for i in range(drugsInSentence):
                    temporalTypeFeatureVector.append(tempType)
                break
        if tempType == "":
            for i in range(drugsInSentence):
                temporalTypeFeatureVector.append("n/a")

    return temporalTypeFeatureVector

def getTenseFeatureVector(fileName, coreNLPClient, drugEvents, raw):
    featureVector = []
    if os.path.isfile('tenseVector_' + fileName.split('.')[0] + '.txt'):
          with open('tenseVector_' + fileName.split('.')[0] + '.txt', 'r') as filehandle:
              for line in filehandle:
                  currentValue = line[:-1]
                  featureVector.append(currentValue)
          return featureVector
    annotatedText = coreNLPClient.annotate(raw)
    nltkSentences = punkt_sent_tokenizer.tokenize(raw)
    sentenceSpans = list(punkt_sent_tokenizer.span_tokenize(raw))
    for i in range(len(nltkSentences)):
        if not any(drugEvent in nltkSentences[i] for drugEvent in drugEvents.keys()):
            continue
        drugEventsSet = drugEvents.keys()
        drugEventIndicesFlatMap = flattenValues(drugEvents)
        drugsInSentence = sum(list(map(lambda drugEventIndex : drugEventIndex >= sentenceSpans[i][0] + 1 and drugEventIndex <= sentenceSpans[i][1] + 1, drugEventIndicesFlatMap)))
        if len(annotatedText.sentence) <= i:
            while drugsInSentence > 0:
                featureVector.append("N/A")
                drugsInSentence-=1
            continue
        constituencyTree = annotatedText.sentence[i].parseTree
        VPnode = getFirstVP(constituencyTree)
        tenseFound = False
        if VPnode is not None:
            drugsInSentence = appendTense(VPnode, drugsInSentence, featureVector)
            if drugsInSentence != 0:
                secondVPnode = getFirstVP(VPnode)
                drugsInSentence = appendTense(secondVPnode, drugsInSentence, featureVector)
        if drugsInSentence != 0:
            while drugsInSentence > 0:
                featureVector.append("N/A")
                drugsInSentence-=1

    with open('tenseVector_' + fileName.split('.')[0] + '.txt', 'w') as filehandle:
          for value in featureVector:
              filehandle.write('%s\n' % value)
    return featureVector

def getSectionFeature(fileName, data_dir, drugEventsStartIndices):
    features = []
    data_dir_features = data_dir + "/section_ids/"
    segmentSpan = {}
    xmldoc = minidom.parse(data_dir_features + os.path.splitext(fileName)[0]+'.xmi')

    segments = xmldoc.getElementsByTagName("textspan:Segment")
    for segment in segments:
        if not segment.getAttribute('preferredText') in segmentSpan:
            segmentSpan[segment.getAttribute('preferredText')] = segment.getAttribute('begin')

    if "admission_date" in segmentSpan:
        admissionSegStart = int(segmentSpan.get("admission_date"))
    else:
        admissionSegStart = float('-inf')

    if "history_present_illness" in segmentSpan:
        historySegStart = int(segmentSpan.get("history_present_illness"))
    else:
        historySegStart = float('inf')

    if "hospital_course" in segmentSpan:
        hospitalSegStart = int(segmentSpan.get("hospital_course"))
    else:
        hospitalSegStart = float('inf')

    for i in range(len(drugEventsStartIndices)):
        if int(drugEventsStartIndices[i]) >= admissionSegStart and int(drugEventsStartIndices[i]) < historySegStart:
            feature = '0'
        elif int(drugEventsStartIndices[i]) >= hospitalSegStart:
            feature = '2'
        elif int(drugEventsStartIndices[i]) >= historySegStart:
            feature = '1'
        else:
            feature = '3'
        features.append(feature)
        # print "word: " + words[i] + " feature: " +feature

    return features

def getTfIdfVectors(drugEvents, raw):
    tfIdfList = []
    nltkSentences = punkt_sent_tokenizer.tokenize(raw)
    sentenceSpans = list(punkt_sent_tokenizer.span_tokenize(raw))
    for i in range(len(nltkSentences)):
        if not any(drugEvent in nltkSentences[i] for drugEvent in drugEvents.keys()):
            continue

        drugEventsSet = drugEvents.keys()
        drugEventIndicesFlatMap = flattenValues(drugEvents)
        drugsInSentence = sum(list(map(lambda drugEventIndex : drugEventIndex >= sentenceSpans[i][0] + 1 and drugEventIndex <= sentenceSpans[i][1] + 1, drugEventIndicesFlatMap)))
        vector = tfidf_vectorizer.transform([nltkSentences[i]])
        npArray = vector.toarray()
        npArrayToList = npArray.tolist()[0]
        for i in range(drugsInSentence):
            tfIdfList.append(npArrayToList)
    return tfIdfList


def getPOSFeature(POSVector):
    posTags = nltk.pos_tag(words)
    posDict = dict((x, y) for x, y in posTags)
    pos_vector = POSVector.transform(posDict)
    tf_idf_vector = tfidf_transformer.fit_transform(pos_vector)
    print(tf_idf_vector.toarray())
    return tf_idf_vector

def getContainsFutureWordsFeature(fileName, drugEvents):
    featureVector = []
    for drug in drugEvents:
        if any(futureWord in drug for futureWord in futureWords):
            for i in drugEvents[drug]:
                futureWord = [futureWord for futureWord in futureWords if(futureWord in drug)][0]
                featureVector.append(futureWord)
        else:
            for i in drugEvents[drug]:
                featureVector.append('n/a')
    return featureVector

def ruleBasedClassifier(data_dir):
    predictedLabels = []
    predictedLabelsInd = 0
    drugEventsFlattened = flattenKeys(drugEvents)
    for i in range(len(drugEventsFlattened)):
        sectionId = sectionsFeatureVector[i]
        # print("length of drug events: " + str(len(drugEvents)) + " length of tense vector: " + str(len(tenseFeatureVector)) + " length of the section feature " + str(len(sectionsFeatureVector)) + "length of future words" + str(len(containsFutureWordsVector)))
        if sectionId == "1" and tenseFeatureVector[i] == "VBD":
            if tenseFeatureVector[i] == "VBG":
                predictedLabels.append("during")
            else:
                predictedLabels.append("before")
        elif FUTURE_WORDS_RULE and sectionId == "2" and containsFutureWordsVector[i] == "1" or tenseFeatureVector[i] == "MD":
            predictedLabels.append("after")
        else:
            predictedLabels.append("during")

    allPredictedLabels.extend(predictedLabels)
    print(file)
    with open('drugClassification.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile)
        for i in range(len(predictedLabels)):
            filewriter.writerow([drugEventsFlattened[i], predictedLabels[i], correctLabels[i]])
        filewriter.writerow(["", "", ""])

def getAllDrugsFromCLAMP(fileName, data_dir, raw):
    drugsFromCLAMP = []
    data_dir_features = data_dir + "/section_ids/"
    xmldoc = minidom.parse(data_dir_features + os.path.splitext(fileName)[0]+'.xmi')
    entities = xmldoc.getElementsByTagName("typesystem:ClampNameEntityUIMA")
    drugsNER = [drug for drug in entities if drug.getAttribute('semanticTag') == 'drug']
    for drug in drugsNER:
        drugsFromCLAMP.append(raw[int(drug.getAttribute('begin')):int(drug.getAttribute('end'))])
    return drugsFromCLAMP

def getDrugEvents(fileName, data_dir, CLAMPdrugs):
    drugEvents = defaultdict(list)
    drugEventsStartIndices = []
    drugEventPolarityFeatureVector = []
    data_dir_features = data_dir + "/treatment_events/"
    f = open(os.path.join(data_dir_features, fileName.split('.')[0] + ".event.xml"), 'r')
    raw = f.read()
    root = ET.fromstring(raw)
    eventStart = root.findall("./EVENT")
    for event in eventStart:
        eventText = event.attrib['text']
        eventWords = eventText.split()
        for word in eventWords:
            if word in CLAMPdrugs:
                drugEvents[eventText].append(int(event.attrib['start']))
                drugEventsStartIndices.append(int(event.attrib['start']))
                drugEventPolarityFeatureVector.append(event.attrib['polarity'])
                break
    return drugEvents, drugEventsStartIndices, drugEventPolarityFeatureVector

def plot(featureVector, labelsList):
    classLabels = ['before', 'during', 'after', 'n/a']
    classColors = ['r', 'y', 'g', 'b']
    for index,label in enumerate(classLabels):
        class_features = featureVector[[i for i in range(len(labelsList)) if labelsList[i] == label]]
        pyplot.scatter(class_features[:, 2], class_features[:, 3], c = classColors[index])
    pyplot.show()
# POSVector = getPOSFeatureVector()
positivePredictions = 0
overallPredictions = 0
allCorrectLabels = []
allPredictedLabels = []

def getFeatureVectorAndLabels(data_dir):
    samplesList = []
    labelsList = []
    tfIdfFeatureVectorList = []
    with open('drugClassification.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(['Drug', 'Predicted Label', 'Correct Label'])

    coreNLPClient = CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], timeout=100000, memory='8G')
    for file in os.listdir(data_dir):
        if not file.endswith('.txt') or not os.path.isfile(os.path.join(data_dir, os.path.splitext(file)[0])):
            continue

        f = open(os.path.join(data_dir, file), 'r')
        raw = f.read()
        CLAMPdrugs = getAllDrugsFromCLAMP(file, data_dir, raw)
        drugEvents, drugEventsStartIndices, drugEventPolarityFeatureVector = getDrugEvents(file, data_dir, CLAMPdrugs)

        correctLabels = getLabels(file, drugEvents, data_dir)
        allCorrectLabels.extend(correctLabels)
        sectionsFeatureVector = getSectionFeature(file, data_dir, drugEventsStartIndices)
        containsFutureWordsVector = getContainsFutureWordsFeature(file, drugEvents)
        tenseFeatureVector = getTenseFeatureVector(file, coreNLPClient, drugEvents, raw)
        temporalTypeFeatureVector = getTemporalCluesFeatureVectors(file, drugEvents, raw, data_dir)
        tfIdfFeatureVectorList += getTfIdfVectors(drugEvents, raw)
        features = [sectionsFeatureVector, containsFutureWordsVector, tenseFeatureVector, temporalTypeFeatureVector, drugEventPolarityFeatureVector]
        for i in range(len(sectionsFeatureVector)):
            sampleList = [feature[i] for feature in features]
            samplesList.append(sampleList)
        for label in correctLabels:
            labelsList.append(label)

        # ruleBasedClassifier()
        # break
    ordinalEncoder = OrdinalEncoder()
    featuresVector = ordinalEncoder.fit_transform(samplesList)
    tfIdfFeatureVector=np.array(tfIdfFeatureVectorList)
    print(tfIdfFeatureVector.shape)
    # np.append(featuresVector, tfIdfFeatureVector, axis = 1)
    featuresVector = np.hstack((featuresVector, tfIdfFeatureVector))
    print(featuresVector.shape)
    labelsVector = np.array(labelsList)
    print(labelsVector.shape)
    return featuresVector, labelsVector
    # return featureVector, labelsVector
    # for i in range(len(allCorrectLabels)):
    #     if allCorrectLabels[i] == allPredictedLabels[i]:
    #         positivePredictions+=1

    # print("Accuracy is ")
    # print((positivePredictions / len(allCorrectLabels)) * 100)
    # labels = ["before","during", "after", "n/a"]
    # print(confusion_matrix(allCorrectLabels, allPredictedLabels, labels=labels))
    # print(classification_report(allCorrectLabels,allPredictedLabels))

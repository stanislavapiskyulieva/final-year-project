import os
import xml.etree.ElementTree as ET
import nltk
import csv
from nltk.tokenize import sent_tokenize, WhitespaceTokenizer, word_tokenize, regexp_tokenize, RegexpTokenizer, PunktSentenceTokenizer
from string import punctuation
from xml.dom import minidom
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from numpy import sum
from extract_labels import getLabels
from sklearn.metrics import confusion_matrix, classification_report
from stanfordnlp.server import CoreNLPClient
import requests


patterns = r'''(?x)
         \w+
        '''

HOST = '127.0.0.1'
PORT = 9000
futureWords = ['further', 'future', 'home', 'follow']
tensePOStags = ['MD', 'VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB']
FUTURE_WORDS_RULE = True
nltkTokenizer = RegexpTokenizer(patterns)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

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
            tenseFound = True
            return

def getTenseFeatureVector(fileName):
    drugEventsIndex = 0
    featureVector = []
    annotatedText = coreNLPClient.annotate(raw)
    nltkSentences = nltk.sent_tokenize(raw)
    for i in range(len(nltkSentences)):
        if drugEventsIndex >= len(drugEvents):
            break
        if drugEvents[drugEventsIndex] not in nltkSentences[i]:
            continue
        drugsInSentence = 0
        while drugEventsIndex < len(drugEvents) and drugEvents[drugEventsIndex] in nltkSentences[i]:
            drugsInSentence+=1
            drugEventsIndex+=1
        if len(annotatedText.sentence) <= i:
            while drugsInSentence > 0:
                featureVector.append("N/A")
                drugsInSentence-=1
            break
        constituencyTree = annotatedText.sentence[i].parseTree
        VPnode = getFirstVP(constituencyTree)
        tenseFound = False
        if VPnode is not None:
            appendTense(VPnode, drugsInSentence, featureVector)
            if not tenseFound:
                secondVPnode = getFirstVP(VPnode)
                appendTense(secondVPnode, drugsInSentence, featureVector)
        if not tenseFound:
            while drugsInSentence > 0:
                featureVector.append("N/A")
                drugsInSentence-=1

    return featureVector

def getSectionFeature(fileName):
    features = []
    data_dir_features = "../data/raw_harvard_tlink/section_ids/"
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

def getPOSFeature(POSVector):
    posTags = nltk.pos_tag(words)
    posDict = dict((x, y) for x, y in posTags)
    pos_vector = POSVector.transform(posDict)
    tf_idf_vector = tfidf_transformer.fit_transform(pos_vector)
    # print(tf_idf_vector.toarray())
    return tf_idf_vector

def getContainsFutureWordsFeature(fileName):
    featureVector = []
    for drug in drugEvents:
        if any(futureWord in drug for futureWord in futureWords):
            featureVector.append('1')
        else:
            featureVector.append('0')
    return featureVector

def ruleBasedClassifier():
    correctLabels = getLabels(file, drugEvents)
    allCorrectLabels.extend(correctLabels)
    predictedLabels = []
    predictedLabelsInd = 0
    for i in range(len(drugEvents)):
        sectionId = sectionsFeatureVector[i]
        if sectionId == "1":
            predictedLabels.append("before")
        elif FUTURE_WORDS_RULE and sectionId == "2" and containsFutureWordsVector[i] == "1":
            predictedLabels.append("after")
        else:
            predictedLabels.append("during")

    allPredictedLabels.extend(predictedLabels)
    print(file)
    with open('drugClassification.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile)
        for i in range(len(predictedLabels)):
            filewriter.writerow([drugEvents[i], predictedLabels[i], correctLabels[i]])
        filewriter.writerow(["", "", ""])

def getAllDrugsFromCLAMP(fileName):
    drugsFromCLAMP = []
    data_dir_features = "../data/raw_harvard_tlink/section_ids/"
    xmldoc = minidom.parse(data_dir_features + os.path.splitext(fileName)[0]+'.xmi')
    drugsNER = xmldoc.getElementsByTagName("typesystem:ClampNameEntityUIMA")
    for drug in drugsNER:
        drugsFromCLAMP.append(raw[int(drug.getAttribute('begin')):int(drug.getAttribute('end'))])
    return drugsFromCLAMP

def getDrugEvents(fileName):
    drugEvents = []
    drugEventsStartIndices = []
    data_dir_features = "../data/raw_harvard_tlink/treatment_events"
    f = open(os.path.join(data_dir_features, fileName.split('.')[0] + ".event.xml"), 'r')
    raw = f.read()
    root = ET.fromstring(raw)
    eventStart = root.findall("./EVENT")
    for event in eventStart:
        eventText = event.attrib['text']
        eventWords = eventText.split()
        for word in eventWords:
            if word in CLAMPdrugs:
                drugEvents.append(eventText)
                drugEventsStartIndices.append(event.attrib['start'])
                break
    return drugEvents, drugEventsStartIndices

data_dir = "../data/raw_harvard_tlink"
# POSVector = getPOSFeatureVector()
positivePredictions = 0
overallPredictions = 0
allCorrectLabels = []
allPredictedLabels = []

with open('drugClassification.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['Drug', 'Predicted Label', 'Correct Label'])

coreNLPClient = CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], timeout=30000, memory='8G')

for file in os.listdir(data_dir):
    if not file.endswith('.txt') or not os.path.isfile(os.path.join(data_dir, os.path.splitext(file)[0])):
        continue

    f = open(os.path.join(data_dir, file), 'r')
    raw = f.read()
    drugsList = []
    CLAMPdrugs = getAllDrugsFromCLAMP(file)
    drugEvents, drugEventsStartIndices = getDrugEvents(file)
    sectionsFeatureVector = getSectionFeature(file)
    containsFutureWordsVector = getContainsFutureWordsFeature(file)
    tenseFeatureVector = getTenseFeatureVector(file)
    # posFeatureVector = getPOSFeature(POSVector)
    # assert len(treatmentsFeatureVector) == len(sectionsFeatureVector) ==  len(drugFeatureVector)

    ruleBasedClassifier()
    # break
for i in range(len(allCorrectLabels)):
    if allCorrectLabels[i] == allPredictedLabels[i]:
        positivePredictions+=1

print("Accuracy is ")
# print((positivePredictions / len(allCorrectLabels)) * 100)
labels = ["before","during", "after", "n/a"]
print(confusion_matrix(allCorrectLabels, allPredictedLabels, labels=labels))
# print(classification_report(allCorrectLabels,allPredictedLabels))

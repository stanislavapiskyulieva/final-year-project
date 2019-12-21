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

patterns = r'''(?x)
         \w+
        '''

futureWords = ['further', 'future', 'home', 'follow']
tokenizer = RegexpTokenizer(patterns)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

def getPOSFeatureVector():
    corpus = {}
    for file in os.listdir(data_dir):
        if not file.endswith('.txt') or not os.path.isfile(os.path.join(data_dir, os.path.splitext(file)[0])):
            continue
        f = open(os.path.join(data_dir, file), 'r')
        raw = f.read()
        words = tokenizer.tokenize(raw)
        posTags = nltk.pos_tag(words)
        posDict = dict((x, y) for x, y in posTags)
        corpus.update(posDict)

    vec = DictVectorizer(sparse=False)
    pos_vector = vec.fit_transform(corpus)
    return vec



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

def ruleBasedClassifier():
    correctLabels = getLabels(file, drugEvents)
    allCorrectLabels.extend(correctLabels)
    predictedLabels = []
    predictedLabelsInd = 0
    for i in range(len(drugEvents)):
        sectionId = sectionsFeatureVector[i]
        # if sectionId == "1" and drugFeatureVector[i] is 'DRUG':
        if sectionId == "1":
            predictedLabels.append("before")
        # elif sectionId == "1" and drugFeatureVector[i] is 'TREATMENT':
            # predictedLabels.append("during")
        elif sectionId == "2" and any(futureWord in drugEvents[i] for futureWord in futureWords):
            predictedLabels.append("after")
        else:
            predictedLabels.append("during")

    allPredictedLabels.extend(predictedLabels)
    print(file)
    with open('drugClassification.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
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
    filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Drug', 'Predicted Label', 'Correct Label'])

for file in os.listdir(data_dir):
    if not file.endswith('.txt') or not os.path.isfile(os.path.join(data_dir, os.path.splitext(file)[0])):
        continue

    f = open(os.path.join(data_dir, file), 'r')
    raw = f.read()
    span_generator = tokenizer.span_tokenize(raw)
    spans = [span for span in span_generator]
    words = tokenizer.tokenize(raw)
    drugsList = []
    CLAMPdrugs = getAllDrugsFromCLAMP(file)
    drugEvents, drugEventsStartIndices = getDrugEvents(file)
    sectionsFeatureVector = getSectionFeature(file)
    # posFeatureVector = getPOSFeature(POSVector)
    # assert len(treatmentsFeatureVector) == len(sectionsFeatureVector) ==  len(drugFeatureVector)

    ruleBasedClassifier()
    # break

for i in range(len(allCorrectLabels)):
    if allCorrectLabels[i] == allPredictedLabels[i]:
        positivePredictions+=1

print("Accuracy is ")
print((positivePredictions / len(allCorrectLabels)) * 100)
labels = ["before","during", "after", "n/a"]
print(confusion_matrix(allCorrectLabels, allPredictedLabels, labels=labels))
# print(classification_report(allCorrectLabels,allPredictedLabels))

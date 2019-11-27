import os
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import sent_tokenize, WhitespaceTokenizer, word_tokenize, regexp_tokenize, RegexpTokenizer, PunktSentenceTokenizer
from string import punctuation
from xml.dom import minidom
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from numpy import sum
from extract_labels import getLabels


patterns = r'''(?x)
         \w+
        '''
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

    for i in range(len(words)):
        if spans[i][0] >= admissionSegStart and spans[i][0] < historySegStart:
            feature = '0'
        elif spans[i][0] >= hospitalSegStart:
            feature = '2'
        elif spans[i][0] >= historySegStart:
            feature = '1'
        else:
            feature = '3'
        features.append(feature)
        # print "word: " + words[i] + " feature: " +feature

    return features

def getStartEndIndices(fileName):
    data_dir_features = "../data/raw_harvard_tlink/treatment_events"
    startIndices = set()
    endIndices = set()
    f = open(os.path.join(data_dir_features, fileName.split('.')[0] + ".event.xml"), 'r')
    raw = f.read()
    root = ET.fromstring(raw)
    eventStart = root.findall("./EVENT")
    for event in eventStart:
        startIndices.add(event.attrib['start'])
        endIndices.add(event.attrib['end'])
    return startIndices, endIndices

def getTreatmentFeature(fileName):
    features = []
    startIndices, endIndices = getStartEndIndices(fileName)
    chunkStart = False
    offset = 0
    for i in range(len(words)):
        startIndex = spans[i][0] + 1
        endIndex = spans[i][1] + 1
        if str(startIndex) in startIndices and str(endIndex) in endIndices:
            feature = "B-TREATMENT"
            chunkStart = False
        elif str(startIndex) in startIndices:
            feature = "B-TREATMENT"
            chunkStart = True
            offset = spans[i][1] - spans[i][0]
        elif str(endIndex) in endIndices:
            feature = "I-TREATMENT"
            chunkStart = False
        elif chunkStart and str((startIndex - offset - 1)) in startIndices:
            feature = "I-TREATMENT"
            offset += spans[i][1] - spans[i][0] - 1
        else:
            feature = "O-TREATMENT"
        features.append(feature)
    return features

def getDrugFeature(fileName):
    features = []
    data_dir_features = "../data/raw_harvard_tlink/section_ids/"
    xmldoc = minidom.parse(data_dir_features + os.path.splitext(fileName)[0]+'.xmi')
    drugsNER = xmldoc.getElementsByTagName("typesystem:ClampNameEntityUIMA")
    drugStartIndices = []
    for drug in drugsNER:
        drugStartIndices.append(drug.getAttribute('begin'))
    for i in range(len(words)):
        startIndex = spans[i][0]
        if treatmentsFeatureVector[i] is 'O-TREATMENT':
            features.append('O')
        elif str(startIndex) in drugStartIndices:
            print("word is " + words[i])
            features.append('DRUG')
        else:
            features.append('TREATMENT')

    return features

def getPOSFeature(POSVector):
    posTags = nltk.pos_tag(words)
    posDict = dict((x, y) for x, y in posTags)
    pos_vector = POSVector.transform(posDict)
    tf_idf_vector = tfidf_transformer.fit_transform(pos_vector)
    # print(tf_idf_vector.toarray())
    return tf_idf_vector

def ruleBasedClassifier():
    correctLabels = getLabels(file)
    predictedLabels = []
    for i in range(len(treatmentsFeatureVector)):
        if not treatmentsFeatureVector[i] == "B-TREATMENT":
            continue
        sectionId = sectionsFeatureVector[i]
        if sectionId == "1" and drugFeatureVector[i] is 'DRUG':
            predictedLabels.append("before")
        elif sectionId == "1" and drugFeatureVector[i] is 'TREATMENT':
            predictedLabels.append("during")
        elif sectionId == "2":
            predictedLabels.append("during")

    correctPredictions = 0
    print(file)
    if len(correctLabels) < len(predictedLabels):
        labelsLen = len(correctLabels)
    else:
        labelsLen = len(predictedLabels)
    for i in range(labelsLen):
        if correctLabels[i] == predictedLabels[i]:
            correctPredictions+=1
        else:
            print("correct label: " + correctLabels[i] + " predicted label: " + predictedLabels[i])

    return correctPredictions, len(correctLabels)


data_dir = "../data/raw_harvard_tlink"
POSVector = getPOSFeatureVector()
positivePredictions = 0
overallPredictions = 0
for file in os.listdir(data_dir):
    if not file.endswith('.txt') or not os.path.isfile(os.path.join(data_dir, os.path.splitext(file)[0])):
        continue

    f = open(os.path.join(data_dir, file), 'r')
    raw = f.read()
    span_generator = tokenizer.span_tokenize(raw)
    spans = [span for span in span_generator]
    words = tokenizer.tokenize(raw)
    treatmentsFeatureVector = getTreatmentFeature(file)
    sectionsFeatureVector = getSectionFeature(file)
    posFeatureVector = getPOSFeature(POSVector)
    drugFeatureVector = getDrugFeature(file)
    assert len(treatmentsFeatureVector) == len(sectionsFeatureVector) ==  len(drugFeatureVector)
    currentPrediction, currentCount = ruleBasedClassifier()
    positivePredictions += currentPrediction
    overallPredictions += currentCount
    # assert len(treatmentsFeatureVector) == len(sectionsFeatureVector) ==  len(posFeatureVector)
    # break

print("Accuracy is ")
print((positivePredictions / overallPredictions) * 100)

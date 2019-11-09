import os
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import sent_tokenize, WhitespaceTokenizer, word_tokenize, regexp_tokenize, RegexpTokenizer, PunktSentenceTokenizer
from string import punctuation
from xml.dom import minidom
from collections import defaultdict

patterns = r'''(?x)
         \w+
        '''
tokenizer = RegexpTokenizer(patterns)

def getSectionFeature(raw, fileName):
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

    span_generator = tokenizer.span_tokenize(raw)
    spans = [span for span in span_generator]
    words = tokenizer.tokenize(raw)
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
    f = open(os.path.join(data_dir_features, os.path.splitext(fileName)[0]), 'r')
    raw = f.read()
    root = ET.fromstring(raw)
    eventStart = root.findall("./EVENT")
    for event in eventStart:
        startIndices.add(event.attrib['start'])
        endIndices.add(event.attrib['end'])
    return startIndices, endIndices

def getTreatmentFeature(raw, fileName):
    features = []
    startIndices, endIndices = getStartEndIndices(file)
    span_generator = tokenizer.span_tokenize(raw)
    spans = [span for span in span_generator]
    words = tokenizer.tokenize(raw)
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

def getPOSFeature(raw):
    tokens = tokenizer.tokenize(raw)
    posTags = nltk.pos_tag(tokens)
    posFeatureVector = []
    for tag in posTags:
        posFeatureVector.append(tag[1])
    return posFeatureVector

data_dir = "../data/raw_harvard_tlink"
for file in os.listdir(data_dir):
    if not file.endswith('.txt') or not os.path.isfile(os.path.join(data_dir, os.path.splitext(file)[0])):
        continue

    f = open(os.path.join(data_dir, file), 'r')
    raw = f.read()
    treatmentsFeatureVector = getTreatmentFeature(raw, file)
    sectionsFeatureVector = getSectionFeature(raw, file)
    posFeatureVector = getPOSFeature(raw)
    assert len(treatmentsFeatureVector) == len(sectionsFeatureVector) ==  len(posFeatureVector)

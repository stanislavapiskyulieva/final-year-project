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

def getSectionLabels(raw, fileName):
    labels = []
    data_dir_labels = "../data/raw_harvard_tlink/section_ids/"
    segmentSpan = {}
    tokenizer = RegexpTokenizer(patterns)
    xmldoc = minidom.parse(data_dir_labels + os.path.splitext(fileName)[0]+'.xmi')
    segments = xmldoc.getElementsByTagName("textspan:Segment")
    for segment in segments:
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
            label = '0'
        elif spans[i][0] >= hospitalSegStart:
            label = '2'
        elif spans[i][0] >= historySegStart:
            label = '1'
        else:
            label = '3'
        labels.append(label)

    return labels


def getStartEndIndices(fileName):
    data_dir_labels = "../data/raw_harvard_tlink/treatment_events"
    startIndices = set()
    endIndices = set()
    f = open(os.path.join(data_dir_labels, os.path.splitext(fileName)[0]), 'r')
    raw = f.read()
    root = ET.fromstring(raw)
    eventStart = root.findall("./EVENT")
    for event in eventStart:
        startIndices.add(event.attrib['start'])
        endIndices.add(event.attrib['end'])
    return startIndices, endIndices

def getTreatmentLabels(raw, fileName):
    labels = []
    startIndices, endIndices = getStartEndIndices(file)
    tokenizer = RegexpTokenizer(patterns)
    span_generator = tokenizer.span_tokenize(raw)
    spans = [span for span in span_generator]
    words = tokenizer.tokenize(raw)
    chunkStart = False
    offset = 0
    for i in range(len(words)):
        startIndex = spans[i][0] + 1
        endIndex = spans[i][1] + 1
        if str(startIndex) in startIndices and str(endIndex) in endIndices:
            label = "B-TREATMENT"
            chunkStart = False
        elif str(startIndex) in startIndices:
            label = "B-TREATMENT"
            chunkStart = True
            offset = spans[i][1] - spans[i][0]
        elif str(endIndex) in endIndices:
            label = "I-TREATMENT"
            chunkStart = False
        elif chunkStart and str((startIndex - offset - 1)) in startIndices:
            label = "I-TREATMENT"
            offset += spans[i][1] - spans[i][0] - 1
        else:
            label = "O-TREATMENT"

        labels.append(label)
    return labels


patterns = r'''(?x)
         \w+
        '''

data_dir = "../data/raw_harvard_tlink"
for file in os.listdir(data_dir):
    if not file.endswith('.txt') or not os.path.isfile(os.path.join(data_dir, os.path.splitext(file)[0])):
        continue

    f = open(os.path.join(data_dir, file), 'r')
    raw = f.read()
    treatmentLabels = getTreatmentLabels(raw, file)
    sectionLabels = getSectionLabels(raw, file)

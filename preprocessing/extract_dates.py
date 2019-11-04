import os
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import sent_tokenize, WhitespaceTokenizer, word_tokenize, regexp_tokenize, RegexpTokenizer
from string import punctuation

patterns = r'''(?x)
        \d+:\d
        | \d+/\d+/\d+
        | \d+-\d+-\d+
        | \d+.\d?
        | \w+
        '''

def getStartEndIndices(fileName):
    data_dir_labels = "../data/raw_harvard_tlink/treatment_events"
    startIndices = set()
    endIndices = set()
    f = open(os.path.join(data_dir_labels, os.path.splitext(fileName)[0]), 'r')
    raw = f.read()
    root = ET.fromstring(raw)
    timex3Start = root.findall("./TIMEX3")
    for time in timex3Start:
        startIndices.add(time.attrib['start'])
        endIndices.add(time.attrib['end'])
    return startIndices, endIndices


# do_not_remove = [':', '-', '\\', '/']
# remove_punctuation = [p for p in punctuation if p not in do_not_remove]
# print remove_punctuation
data_dir = "../data/raw_harvard_tlink"
labels = []
for file in os.listdir(data_dir):
    if not file.endswith('.txt'):
        continue

    startIndices, endIndices = getStartEndIndices(file)

    f = open(os.path.join(data_dir, file), 'r')
    raw = f.read()
    tokenizer = RegexpTokenizer(patterns)
    span_generator = tokenizer.span_tokenize(raw.strip())
    spans = [span for span in span_generator]
    # tokenizer = RegexpTokenizer(patterns)
    words = tokenizer.tokenize(raw)
    chunkStart = False
    offset = 0
    for i in range(len(words)):
        startIndex = spans[i][0] + 1
        endIndex = spans[i][1] + 1
        if str(startIndex) in startIndices and str(endIndex) in endIndices:
            label = "B-TIMEX3"
            chunkStart = False
        elif str(startIndex) in startIndices:
            label = "B-TIMEX3"
            chunkStart = True
            offset = len(words[i])
        elif str(endIndex) in endIndices:
            label = "I-TIMEX3"
            chunkStart = False
        elif chunkStart and str((startIndex - offset - 1)) in startIndices:
            label = "I-TIMEX3"
            offset += len(words[i]) + 1
        else:
            label = "O-TIMEX3"

        labels.append(label)
        print "word:" + words[i] + " label: " + label
    print file

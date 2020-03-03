import os
import xml.etree.ElementTree as ET
import nltk
import csv
import re
import math
import sys
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
from gensim.models import Word2Vec,KeyedVectors

patterns = r'''(?x)
         \w*follow-up\w*
         |
         \w+
        '''

data_dir = "../data/training_data"
futureWords = ['further', 'future ', 'home ', 'follow ', 'follow-up', 'discharged on', 'discharged home on', 'discharged to home on', 'discharge ', 'recommend ', 'recommended', 'prescribe ', 'prescription', 'prescribed', 'will ', ' continue ', 'be given', 'may ']
tensePOStags = ['MD', 'VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB']
FUTURE_WORDS_RULE = True
WINDOW_SIZE = 9
WINDOW_SIZE_EMBEDDING = 3
DIMENSIONS = 300
nltkTokenizer = RegexpTokenizer(patterns)
tfidf_vectorizer=TfidfVectorizer()#ngram_range = (1,2))
training_text = ""
corpus = []
for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            f = open(os.path.join(data_dir, file), 'r')
            raw = f.read()
            training_text += raw
            corpus.append(raw.lower())
punkt_sent_tokenizer = PunktSentenceTokenizer(training_text)
tfidf_vectorizer.fit(corpus)
drugToAliasDict = {}
aliasToDrugDict = {}
word2VecModel = KeyedVectors.load_word2vec_format('glove.6B.300d.txt.word2vec', binary=False)

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
    drugEventsList = []
    for drug in drugEvents:
        for i in drugEvents[drug]:
            drugEventsList.append(drug)
            drugEventIndicesFlatMap.append(i)
    return drugEventsList, drugEventIndicesFlatMap

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
        drugEventsList, drugEventIndicesFlatMap = flattenValues(drugEvents)
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
        sentence = nltkSentences[i]#.replace('-', ' ').replace('#', '').replace('(', '').replace(')', '').replace('%', ' ').replace('\'', '').replace(',', ' ').replace('.', ' ')
        if not any(drugEvent in sentence for drugEvent in drugEvents.keys()):
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

    drugEventsStartIndices.sort()
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

def getPositionInTextFeatureVector(raw, drugEvents):
    positionInTextFeatureVector = []

    nltkSentences = punkt_sent_tokenizer.tokenize(raw)
    sentenceSpans = list(punkt_sent_tokenizer.span_tokenize(raw))

    drugEventsSet = drugEvents.keys()
    drugEventsList, drugEventIndicesFlatMap = flattenValues(drugEvents)
    for i in range(len(nltkSentences)):
        if not any(drugEvent in nltkSentences[i] for drugEvent in drugEvents.keys()):
            continue

        drugsInSentence = sum(list(map(lambda drugEventIndex : drugEventIndex >= sentenceSpans[i][0] + 1 and drugEventIndex <= sentenceSpans[i][1] + 1, drugEventIndicesFlatMap)))

        normalisedPosition = i / len(nltkSentences)
        for i in range(drugsInSentence):
            positionInTextFeatureVector.append(normalisedPosition)
    return positionInTextFeatureVector

def getWordEmbeddingsFeatureVector(raw, drugEvents, problem):
    drugEventsSet = sorted(list(drugEvents.keys()), reverse= True, key = len)
    drugOccurences = {}
    wordEmbeddingsFeatureVector = []
    for drug in drugEventsSet:
        raw = re.sub(r"\b%s\b" % drug, drugToAliasDict[drug], raw)
        drugOccurences.update({drug: 0})
    tokens = nltkTokenizer.tokenize(raw)
    spans = list(nltkTokenizer.span_tokenize(raw))
    for i in range(len(tokens)):
        if tokens[i].startswith("alias"):
            drug = aliasToDrugDict[tokens[i]]
            if drugOccurences[drug] >= len(drugEvents[drug]):
                continue
            drugOccurences.update({drug: drugOccurences[drug] + 1})
            words = drug.split()
            if len(words) == 1:
                wordEmbedding = getWordEmbeddingsWindowVectorFront(tokens, i, int((WINDOW_SIZE_EMBEDDING - 1) / 2))
                try:
                    wordEmbedding += word2VecModel[words[0].lower()]
                except KeyError:
                    wordEmbedding += np.zeros(DIMENSIONS)
                    problem += 1
                wordEmbedding += getWordEmbeddingsWindowVectorBack(tokens, i, int((WINDOW_SIZE_EMBEDDING - 1) / 2))
                wordEmbeddingsFeatureVector.append(wordEmbedding / WINDOW_SIZE_EMBEDDING)
            else:
                 wordLength = len(words)
                 n1 = int((WINDOW_SIZE_EMBEDDING - wordLength) / 2)
                 if(wordLength % 2 == 0):
                     n2 = int((WINDOW_SIZE_EMBEDDING - wordLength) / 2) + 1
                 else:
                      n2 = n1
                 wordEmbedding = getWordEmbeddingsWindowVectorFront(tokens, i, n1)
                 for i in range(wordLength):
                     try:
                         wordEmbedding += word2VecModel[words[i].lower()]
                     except KeyError:
                         wordEmbedding += np.zeros(DIMENSIONS)
                         problem += 1
                 wordEmbedding += getWordEmbeddingsWindowVectorBack(tokens, i, n2)
                 wordEmbeddingsFeatureVector.append(wordEmbedding / WINDOW_SIZE_EMBEDDING)
    for occurence in drugOccurences.keys():
        if(drugOccurences[occurence] == 0):
            wordEmbeddingsFeatureVector.append(np.zeros(DIMENSIONS))
    return wordEmbeddingsFeatureVector, problem

def getTfIdfVectors(drugEvents, raw, drugEventsStartIndices):
    drugEventsSet = sorted(list(drugEvents.keys()), reverse= True, key = len)
    drugOccurences = {}
    tfIdfList = []
    for drug in drugEventsSet:
        raw = re.sub(r"\b%s\b" % drug, drugToAliasDict[drug], raw)
        drugOccurences.update({drug: 0})
    raw = raw.lower()
    tokens = nltkTokenizer.tokenize(raw)
    spans = list(nltkTokenizer.span_tokenize(raw))
    for i in range(len(tokens)):
        if tokens[i].startswith("alias"):
            drug = aliasToDrugDict[tokens[i]]
            if drugOccurences[drug] >= len(drugEvents[drug]):
                continue
            drugOccurences.update({drug: drugOccurences[drug] + 1})
            words = drug.split()
            if len(words) == 1:
                windowList = []
                windowList = windowList + getNGramFront(tokens, i, int((WINDOW_SIZE - 1) / 2))
                windowList.append(words[0])
                windowList = windowList + getNGramBack(tokens, i, int((WINDOW_SIZE - 1) / 2))
                str = " ". join(windowList)
                vector = tfidf_vectorizer.transform([str])
                npArray = vector.toarray()
                npArrayToList = npArray.tolist()[0]
                tfIdfList.append(npArrayToList)
            else:
                 wordLength = len(words)
                 n1 = int((WINDOW_SIZE - wordLength) / 2)
                 if(wordLength % 2 == 0):
                     n2 = int((WINDOW_SIZE - wordLength) / 2) + 1
                 else:
                      n2 = n1
                 windowList = []
                 windowList = windowList + getNGramFront(tokens, i, n1)
                 for i in range(wordLength):
                     windowList.append(words[i])
                 windowList = windowList + getNGramBack(tokens, i, n2)
                 str = " ". join(windowList)
                 vector = tfidf_vectorizer.transform([str])
                 npArray = vector.toarray()
                 npArrayToList = npArray.tolist()[0]
                 tfIdfList.append(npArrayToList)
    for occurence in drugOccurences.keys():
        if(drugOccurences[occurence] == 0):
            vector = tfidf_vectorizer.transform(["n/a"])
            npArray = vector.toarray()
            npArrayToList = npArray.tolist()[0]
            tfIdfList.append(npArrayToList)
    return tfIdfList

def getNGramFront(tokens, index, n):
    windowList = []
    startInd = index - n
    for i in range(n):
        if(startInd >= len(tokens)): windowList.append("n/a")
        else: windowList.append(tokens[startInd])
        startInd += 1
    return windowList

def getNGramBack(tokens, index, n):
    windowList = []
    startInd = index + 1
    for i in range(n):
        if(startInd >= len(tokens)): windowList.append("n/a")
        else: windowList.append(tokens[startInd])
        startInd += 1
    return windowList

def getWordEmbeddingsWindowVectorFront(tokens, index, n):
    wordEmbedding = np.zeros(DIMENSIONS)
    startInd = index - n
    for i in range(n):
        if(startInd >= len(tokens)): wordEmbedding += np.zeros(DIMENSIONS)
        else:
            try:
                wordEmbedding += word2VecModel[tokens[startInd].lower()]
            except KeyError:
                wordEmbedding += np.zeros(DIMENSIONS)
        startInd += 1
    return wordEmbedding

def getWordEmbeddingsWindowVectorBack(tokens, index, n):
    wordEmbedding = np.zeros(DIMENSIONS)
    startInd = index + 1
    for i in range(n):
        if(startInd >= len(tokens)): wordEmbedding += np.zeros(DIMENSIONS)
        else:
            try:
                wordEmbedding += word2VecModel[tokens[startInd].lower()]
            except KeyError:
                wordEmbedding += np.zeros(DIMENSIONS)
        startInd += 1
    return wordEmbedding

def getContainsFutureWordsFeature(raw, drugEvents):
    containsFutureWordsFeatureVector = []
    proximityToFutureWordFeatureVector = []
    futureWordsCountFeatureVector = []
    drugEventsSet = drugEvents.keys()
    drugEventsList, drugEventIndicesFlatMap = flattenValues(drugEvents)

    nltkSentences = punkt_sent_tokenizer.tokenize(raw)
    sentenceSpans = list(punkt_sent_tokenizer.span_tokenize(raw))

    for i in range(len(nltkSentences)):
        if not any(drugEvent in nltkSentences[i] for drugEvent in drugEventsSet):
            continue
        for index, drug in enumerate(drugEventsList):
            if drugEventIndicesFlatMap[index] >= sentenceSpans[i][0] and drugEventIndicesFlatMap[index] <= sentenceSpans[i][1] + 1:
                if any(futureWord in nltkSentences[i] for futureWord in futureWords):
                    futureWordsInSen = [futureWord for futureWord in futureWords if(futureWord in nltkSentences[i])]
                    futureWordsCountFeatureVector.append(len(futureWordsInSen))
                    distanceFromFutureWord = []
                    tokensInSen = nltkTokenizer.tokenize(nltkSentences[i])
                    drugIndex = tokensInSen.index(drug.replace("-", " ").split()[0])
                    for futureWord in futureWordsInSen:
                        if futureWord in drug:
                            distanceFromFutureWord.append(0)
                        distanceFromFutureWord.append(math.fabs(drugIndex - tokensInSen.index(futureWord.split()[0])))
                    minimumDistance = min(distanceFromFutureWord)
                    proximityToFutureWordFeatureVector.append(minimumDistance)
                    containsFutureWordsFeatureVector.append(futureWordsInSen[distanceFromFutureWord.index(minimumDistance)])
                else:
                    containsFutureWordsFeatureVector.append('n/a')
                    futureWordsCountFeatureVector.append(0)
                    proximityToFutureWordFeatureVector.append(sys.maxsize * 2)
    return containsFutureWordsFeatureVector, proximityToFutureWordFeatureVector, futureWordsCountFeatureVector

def ruleBasedClassifier(data_dir, allPredictedLabels, drugEvents, sectionsFeatureVector, tenseFeatureVector, containsFutureWordsVector):
    predictedLabels = []
    predictedLabelsInd = 0
    drugEventsFlattened = flattenKeys(drugEvents)
    for i in range(len(drugEventsFlattened)):
        sectionId = sectionsFeatureVector[i]
        if sectionId == "1" and tenseFeatureVector[i] == "VBD":
            predictedLabels.append("before")
        elif FUTURE_WORDS_RULE and sectionId == "2" and containsFutureWordsVector[i] != "n/a" or tenseFeatureVector[i] == "MD":
            predictedLabels.append("after")
        else:
            predictedLabels.append("during")

    allPredictedLabels.extend(predictedLabels)
    # with open('drugClassification.csv', 'a') as csvfile:
    #     filewriter = csv.writer(csvfile)
    #     for i in range(len(predictedLabels)):
    #         filewriter.writerow([drugEventsFlattened[i], predictedLabels[i], correctLabels[i]])
    #     filewriter.writerow(["", "", ""])

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
    drugEventModalityFeatureVector = []
    data_dir_features = data_dir + "/treatment_events/"
    f = open(os.path.join(data_dir_features, fileName.split('.')[0] + ".event.xml"), 'r')
    raw = f.read()
    root = ET.fromstring(raw)
    eventStart = root.findall("./EVENT")
    for event in eventStart:
        eventText = event.attrib['text']#.replace('-',' ').replace('#', '').replace('(', '').replace(')', '').replace('%', ' ').replace('\'', '').replace(',', ' ').replace('.', ' ')
        eventWords = eventText.split()#.replace('-', ' ').replace('#', '').replace('(', '').replace(')', '').replace('%', ' ').replace('\'', '').replace(',', ' ').replace('.', ' ').split()
        for word in eventWords:
            if word in CLAMPdrugs:
                if eventText not in drugToAliasDict:
                    alias = 'alias' + str(len(drugToAliasDict))
                    drugToAliasDict.update({eventText: alias})
                    aliasToDrugDict.update({alias: eventText})
                drugEvents[eventText].append(int(event.attrib['start']))
                drugEventsStartIndices.append(int(event.attrib['start']))
                drugEventPolarityFeatureVector.append(event.attrib['polarity'])
                drugEventModalityFeatureVector.append(event.attrib['modality'])
                break
    return drugEvents, drugEventsStartIndices, drugEventPolarityFeatureVector, drugEventModalityFeatureVector

def plot(featureVector, labelsList):
    classLabels = ['before', 'during', 'after', 'n/a']
    classColors = ['r', 'y', 'g', 'b']
    for index,label in enumerate(classLabels):
        class_features = featureVector[[i for i in range(len(labelsList)) if labelsList[i] == label]]
        pyplot.scatter(class_features[:, 2], class_features[:, 3], c = classColors[index])
    pyplot.show()



def getFeatureVectorAndLabels(data_dir):
    positivePredictions = 0
    overallPredictions = 0
    allCorrectLabels = []
    allPredictedLabels = []
    problem = 0
    samplesList = []
    labelsList = []
    tfIdfFeatureVectorList = []
    wordEmbeddingsFeatureVectorList = []
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
        drugEvents, drugEventsStartIndices, drugEventPolarityFeatureVector, drugEventModalityFeatureVector = getDrugEvents(file, data_dir, CLAMPdrugs)

        correctLabels = getLabels(file, drugEvents, data_dir)
        allCorrectLabels.extend(correctLabels)
        sectionsFeatureVector = getSectionFeature(file, data_dir, drugEventsStartIndices)
        containsFutureWordsVector, proximityToFutureWordFeatureVector, futureWordsCountFeatureVector = getContainsFutureWordsFeature(raw, drugEvents)
        tenseFeatureVector = getTenseFeatureVector(file, coreNLPClient, drugEvents, raw)
        temporalTypeFeatureVector = getTemporalCluesFeatureVectors(file, drugEvents, raw, data_dir)
        positionInTextFeatureVector = getPositionInTextFeatureVector(raw, drugEvents)
        wordEmbeddingsFeatureVector, problem = getWordEmbeddingsFeatureVector(raw, drugEvents, problem)
        print(file)
        tfIdfFeatureVector = getTfIdfVectors(drugEvents, raw, drugEventsStartIndices)
        ruleBasedClassifier(data_dir, allPredictedLabels, drugEvents, sectionsFeatureVector, tenseFeatureVector, containsFutureWordsVector)
        if len(tfIdfFeatureVector) != len(sectionsFeatureVector):
            continue
        wordEmbeddingsFeatureVectorList += wordEmbeddingsFeatureVector
        tfIdfFeatureVectorList += tfIdfFeatureVector
        features = [sectionsFeatureVector, containsFutureWordsVector, tenseFeatureVector, temporalTypeFeatureVector, drugEventPolarityFeatureVector, positionInTextFeatureVector, drugEventModalityFeatureVector, proximityToFutureWordFeatureVector, futureWordsCountFeatureVector]
        for i in range(len(sectionsFeatureVector)):
            sampleList = [feature[i] for feature in features]
            samplesList.append(sampleList)
        for label in correctLabels:
            labelsList.append(label)

        # break
    ordinalEncoder = OrdinalEncoder()
    featuresVector = ordinalEncoder.fit_transform(samplesList)
    wordEmbeddingsFeatureVector = np.array(wordEmbeddingsFeatureVectorList)
    print(wordEmbeddingsFeatureVector.shape)
    featuresVector = np.hstack((featuresVector, wordEmbeddingsFeatureVector))
    tfIdfFeatureVector=np.array(tfIdfFeatureVectorList)
    featuresVector = np.hstack((featuresVector, tfIdfFeatureVector))
    print(featuresVector.shape)
    labelsVector = np.array(labelsList)
    print(labelsVector.shape)
    for i in range(len(allCorrectLabels)):
        if allCorrectLabels[i] == allPredictedLabels[i]:
            positivePredictions+=1

    print("Accuracy is ")
    print((positivePredictions / len(allCorrectLabels)) * 100)
    labels = ["before","during", "after"]
    print(confusion_matrix(allCorrectLabels, allPredictedLabels, labels=labels))
    print(classification_report(allCorrectLabels,allPredictedLabels))
    return featuresVector, labelsVector

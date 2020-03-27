from sklearn.svm import SVC
from sklearn import preprocessing
from extract_drugs import getFeatureVectorAndLabels, getAllDrugsFromCLAMP, getDrugEvents, flattenKeys
from joblib import dump,load
import numpy as np
import os
from shutil import copyfile
from collections import Counter
from extract_labels import getLabels


data_dir = "../data/cross_val"
files, drugs, features, X, y = getFeatureVectorAndLabels(data_dir)
trainingSampleCount = 0.8 * len(drugs)
counter = Counter(y)
beforeSampleCount = 0.8 * counter["before"]
afterSampleCount = 0.8 * counter["after"]
filesToProcess = [file for file in os.listdir(data_dir) if (file.endswith('.txt'))]

beforeSamples = 0
afterSamples = 0
for file in filesToProcess:
    f = open(os.path.join(data_dir, file), 'r')
    raw = f.read()
    CLAMPdrugs = getAllDrugsFromCLAMP(file, data_dir, raw)
    drugEvents, drugEventsStartIndices, drugEventPolarityFeatureVector, drugEventModalityFeatureVector = getDrugEvents(file, data_dir, CLAMPdrugs)
    labelsCntr = Counter(getLabels(file, drugEvents, data_dir))
    beforeSamples += labelsCntr["before"]
    afterSamples += labelsCntr["after"]
    if beforeSamples > beforeSampleCount and afterSamples > afterSampleCount:
        print(beforeSamples)
        print(afterSamples)
        copyfile(os.path.join(data_dir, file) ,data_dir + "/test_data/" + file)
        copyfile(os.path.join(data_dir, os.path.splitext(file)[0]),data_dir + "/test_data/" + os.path.splitext(file)[0])
        copyfile(os.path.join(data_dir + '/treatment_events/', file.split('.')[0] + ".event.xml"),data_dir + "/test_data/treatment_events/" + file.split('.')[0] + ".event.xml")
        copyfile(os.path.join(data_dir + '/treatment_events/', file.split('.')[0] + ".sectime.xml"),data_dir + "/test_data/treatment_events/" + file.split('.')[0] + ".sectime.xml")
        copyfile(os.path.join(data_dir + '/treatment_events/', file.split('.')[0] + ".tlink.xml"),data_dir + "/test_data/treatment_events/" + file.split('.')[0] + ".tlink.xml")
        copyfile(os.path.join(data_dir + '/section_ids/', os.path.splitext(file)[0]+'.xmi'),data_dir + "/test_data/section_ids/" + os.path.splitext(file)[0]+'.xmi')
    else:
        copyfile(os.path.join(data_dir, file) ,data_dir + "/training_data/" + file)
        copyfile(os.path.join(data_dir, os.path.splitext(file)[0]),data_dir + "/training_data/" + os.path.splitext(file)[0])
        copyfile(os.path.join(data_dir + '/treatment_events/', file.split('.')[0] + ".event.xml"),data_dir + "/training_data/treatment_events/" + file.split('.')[0] + ".event.xml")
        copyfile(os.path.join(data_dir + '/treatment_events/', file.split('.')[0] + ".sectime.xml"),data_dir + "/training_data/treatment_events/" + file.split('.')[0] + ".sectime.xml")
        copyfile(os.path.join(data_dir + '/treatment_events/', file.split('.')[0] + ".tlink.xml"),data_dir + "/training_data/treatment_events/" + file.split('.')[0] + ".tlink.xml")
        copyfile(os.path.join(data_dir + '/section_ids/', os.path.splitext(file)[0]+'.xmi'),data_dir + "/training_data/section_ids/" + os.path.splitext(file)[0]+'.xmi')

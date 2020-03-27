from sklearn.svm import SVC
from sklearn import preprocessing
from extract_drugs import getFeatureVectorAndLabels
from joblib import dump,load
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

FOLDS = 5

np.set_printoptions(threshold=np.inf)
data_dir = "../data/training_data"
files, drugs, features, X, y = getFeatureVectorAndLabels(data_dir)

labelEncoder = preprocessing.LabelEncoder()
y_encoded = labelEncoder.fit_transform(y)

SVMclassifier = SVC(kernel = 'rbf',  class_weight='balanced', C = 1000.0, gamma = 'auto')
SVMclassifier.fit(X, y_encoded)
dump(SVMclassifier, 'svmModelC.joblib')

from sklearn.svm import SVC
from sklearn import preprocessing
from extract_drugs import getFeatureVectorAndLabels
from joblib import dump,load
import numpy as np

np.set_printoptions(threshold=np.inf)
data_dir = "../data/training_data"
X_train, y_train = getFeatureVectorAndLabels(data_dir)
labelEncoder = preprocessing.LabelEncoder()
y_train_encoded = labelEncoder.fit_transform(y_train)
SVMclassifier = SVC(kernel = 'rbf',  class_weight='balanced', C = 100.0, random_state = 0)
SVMclassifier.fit(X_train, y_train_encoded)
dump(SVMclassifier, 'svmModelC.joblib')

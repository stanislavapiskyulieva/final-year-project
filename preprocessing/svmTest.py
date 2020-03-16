from sklearn.svm import SVC
from sklearn import preprocessing
from extract_drugs import getFeatureVectorAndLabels
from joblib import dump,load
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

np.set_printoptions(threshold=np.inf)

data_dir = "../data/test_data"
sentences, drugs, features, X, y = getFeatureVectorAndLabels(data_dir)
svmModel = load('svmModelC.joblib')

labelEncoder = preprocessing.LabelEncoder()
y_encoded = labelEncoder.fit_transform(y)

y_pred = svmModel.predict(X)

print(confusion_matrix(y_encoded,y_pred))
print(classification_report(y_encoded,y_pred))

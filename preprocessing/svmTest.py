from sklearn.svm import SVC
from sklearn import preprocessing
from extract_drugs import getFeatureVectorAndLabels
from joblib import dump,load
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

np.set_printoptions(threshold=np.inf)
data_dir = "../data/test_data"
X_test, y_test = getFeatureVectorAndLabels(data_dir)
svmModel = load('svmModelC.joblib')
y_pred = svmModel.predict(X_test)
labelEncoder = preprocessing.LabelEncoder()
y_test_encoded = labelEncoder.fit_transform(y_test)
print(X_test)
print(y_pred)
print(y_test_encoded)
print(confusion_matrix(y_test_encoded,y_pred))
print(classification_report(y_test_encoded,y_pred))

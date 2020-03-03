from sklearn.svm import SVC
from sklearn import preprocessing
from extract_drugs import getFeatureVectorAndLabels
from sklearn.model_selection import GridSearchCV
import pickle


FOLDS = 10
param_grid = [
  {'C': [1, 10, 30, 50, 70, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf'], 'class_weight': ['balanced']},
 ]

data_dir = "../data/cross_val"
X, y = getFeatureVectorAndLabels(data_dir)
labelEncoder = preprocessing.LabelEncoder()
y_encoded = labelEncoder.fit_transform(y)
svc = SVC()
clf = GridSearchCV(svc, param_grid, cv=FOLDS)
clf.fit(X, y_encoded)
with open('cvResults.p', 'wb') as fp:
    pickle.dump(clf.cv_results_, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open('bestParams.p', 'wb') as fp:
    pickle.dump(clf.best_params_, fp, protocol=pickle.HIGHEST_PROTOCOL)

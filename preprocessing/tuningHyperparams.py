from sklearn.svm import SVC
from sklearn import preprocessing
from extract_drugs import getFeatureVectorAndLabels
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pickle
from sklearn.metrics import precision_score, make_scorer, f1_score

FOLDS = 8
param_grid = [
  {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 'auto'], 'kernel': ['rbf'], 'class_weight': ['balanced']},
 ]

f1_scorer = make_scorer(f1_score, average='weighted')
data_dir = "../data/training_data"
files, drugs, features, X, y = getFeatureVectorAndLabels(data_dir)
labelEncoder = preprocessing.LabelEncoder()
y_encoded = labelEncoder.fit_transform(y)
svc = SVC()
clf = GridSearchCV(svc, param_grid, cv=FOLDS, n_jobs = -1, verbose=1, scoring=f1_scorer)
clf.fit(X, y_encoded)
with open('cvResults.p', 'wb') as fp:
    pickle.dump(clf.cv_results_, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open('bestParams.p', 'wb') as fp:
    pickle.dump(clf.best_params_, fp, protocol=pickle.HIGHEST_PROTOCOL)

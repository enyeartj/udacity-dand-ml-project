#!/usr/bin/python

import sys
import time
import pickle
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
sys.path.append("../tools/")

from get_words_from_emails import get_word_features
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

def my_score(y, y_pred):
    '''Custom scoring function to use with GridSearchCV so that parameters
    get optomized according to project specifications of both precision and
    recall above 0.3. So if that requirement isn't met, the function returns
    a zero, otherwise it returns the F1 score.
    '''
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    if precision > 0.3 and recall > 0.3:
        return f1
    else:
        return 0.0

my_scorer = make_scorer(my_score)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
    'poi',
    'salary',
    'bonus',
    'deferral_payments', #
    'deferred_income',
#    'director_fees', # only 16 out of 144 data points are non-zero
#    'email_address',
    'exercised_stock_options',
    'expenses',
#    'from_messages',
#    'from_poi_to_this_person',
#    'from_this_person_to_poi',
#    'loan_advances', # only 3 non-zero data points!
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred', #
    'shared_receipt_with_poi',
#    'to_messages',
    'total_payments',
    'total_stock_value',
    'bonus_per_salary', # new!
    'long_term_incentive_per_salary', # new!
    'important_to', # new!
    'important_from', # new!
]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
if False: # create new features and remove outliers
    my_dataset = dict()
    remove_me = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    for key in sorted(data_dict.keys()):
        if key not in remove_me:
            my_dataset[key] = data_dict[key]
            if      (
                        my_dataset[key]['salary'] == 0 or
                        my_dataset[key]['salary'] == 'NaN' or
                        my_dataset[key]['bonus'] == 'NaN' or
                        my_dataset[key]['long_term_incentive'] == 'NaN'
                    ):
                my_dataset[key]['bonus_per_salary'] = 0
                my_dataset[key]['long_term_incentive_per_salary'] = 0
            else:
                my_dataset[key]['bonus_per_salary'] = float(my_dataset[key]['bonus']) / my_dataset[key]['salary']
                my_dataset[key]['long_term_incentive_per_salary'] = float(my_dataset[key]['long_term_incentive']) / my_dataset[key]['salary']
    
    # add important word features
    get_word_features(my_dataset, 'all', 'all', True)
else: # load already dataset with outliers already removed and features added
    DATASET_PICKLE_FILENAME = "my_dataset.pkl"
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        my_dataset = pickle.load(dataset_infile)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features = np.array(features)
labels = np.array(labels)

if False: # select k best with scaled features
    k = 10
    features = MaxAbsScaler().fit_transform(features)
    scaled_kbest = SelectKBest(f_classif, k=k)
    features = scaled_kbest.fit_transform(features, labels)
    print "\n**Scaled** features selected by SelectKBest with k = %d:" % k
    for f_name, selected in zip(features_list[1:], scaled_kbest.get_support()):
        if selected:
            print "    %s" % f_name

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
if False: # Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

if False: # SVM
    from sklearn.svm import SVC
    pipeline_steps = [
        ('scaler', MaxAbsScaler()),
        ('kbest', SelectKBest(f_classif)),
        ('svm', SVC()),
    ]
    clf = Pipeline(pipeline_steps)
    
    search_params = {
        'kbest__k' : [10, 15, 'all'],
        'svm__kernel' : ['rbf', 'linear'],
        'svm__gamma' : [1e-2, 1e-1, 1, 10],
        'svm__C' : [.1, 1, 10, 100]
    }
    gscv = StratifiedShuffleSplit(labels, random_state = 42)
    clf = GridSearchCV(clf, search_params, scoring=my_scorer, cv=gscv, verbose=1)
    clf.fit(features, labels)
    clf = clf.best_estimator_

if False: # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    search_params = {
        'criterion' : ['gini', 'entropy'],
        'splitter' : ['best', 'random']
    }
    clf = DecisionTreeClassifier()
    gscv = StratifiedShuffleSplit(labels, random_state = 42)
    clf = GridSearchCV(clf, search_params, scoring=my_scorer, cv=gscv, verbose=1)
    clf.fit(features, labels)
    clf = clf.best_estimator_

if True: # AdaBoost
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    pipeline_steps = [
        ('scaler', MaxAbsScaler()),
        ('kbest', SelectKBest(f_classif)),
        ('pca', PCA()),
        ('ada', AdaBoostClassifier())
    ]
    clf = Pipeline(pipeline_steps)
    
    search_params = {
        'kbest__k' : [10, 15, 'all'],
        'ada__base_estimator' : [
            DecisionTreeClassifier(),
            DecisionTreeClassifier(max_depth=5),
            DecisionTreeClassifier(max_depth=10),
        ],
        'ada__n_estimators' : [50, 100, 200]
    }
    gscv = StratifiedShuffleSplit(labels, random_state = 42)
    clf = GridSearchCV(clf, search_params, scoring='recall', cv=gscv, verbose=1)
    clf.fit(features, labels)
    clf = clf.best_estimator_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
if False:
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    clf.fit(features_train, labels_train)

if True: # Stratified K-Fold CV
    success_f1 = 0.0
    k_fold = StratifiedKFold(labels, 3, shuffle=True, random_state=42)
    t0 = time.time()
    for k, (train, test) in enumerate(k_fold):
        features_train = features[train]
        features_test = features[test]
        labels_train = labels[train]
        labels_test = labels[test]
#        gscv = StratifiedShuffleSplit(labels_train, random_state = 42)
#        clf = GridSearchCV(clf, search_params, scoring=my_scorer, cv=gscv, verbose=1)
        clf.fit(features_train, labels_train)
#        clf = clf.best_estimator_
        accuracy = clf.score(features_test, labels_test)
        predictions = clf.predict(features_test)
        precision = precision_score(labels_test, predictions)
        recall = recall_score(labels_test, predictions)
        f1 = f1_score(labels_test, predictions)
        print("Fold {0} -- accuracy: {1:.5f}, precision: {2:.5f}, recall: {3:.5f}, F1 score: {4:.5f}".
              format(k, accuracy, precision, recall, f1))
#        if precision > 0.3 and recall > 0.3:
#            print "\nSuccessful Classifier Found!!!"
#            print "Successful Parameters:\n\n", clf.get_params()
#            if f1 > success_f1:
#                print "\nSaving Classifier...\n"
#                dump_classifier_and_data(clf, my_dataset, features_list)
    print "K-Fold Cross-Validation took {0:.3f} seconds".format(time.time()-t0)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
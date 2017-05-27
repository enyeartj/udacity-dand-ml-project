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

if __name__ == '__main__':
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
        'director_fees', # only 16 out of 144 data points are non-zero
    #    'email_address',
        'exercised_stock_options',
        'expenses',
        'from_messages', # redundant/noisy after adding important_to & important_from
        'from_poi_to_this_person', # redundant/noisy after adding important_to & important_from
        'from_this_person_to_poi', # redundant/noisy after adding important_to & important_from
        'loan_advances', # only 3 non-zero data points!
        'long_term_incentive',
        'other',
        'restricted_stock',
        'restricted_stock_deferred', #
        'shared_receipt_with_poi',
        'to_messages', # redundant/noisy after adding important_to & important_from
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
        k = 4
        #features = MaxAbsScaler().fit_transform(features)
        scaled_kbest = SelectKBest(f_classif, k=k)
        features = scaled_kbest.fit_transform(features, labels)
        #print "\n**Scaled** features selected by SelectKBest with k = %d:" % k
        print "\n**Unscaled** features selected by SelectKBest with k = %d:" % k
        for f_name, selected in zip(features_list[1:], scaled_kbest.get_support()):
            if selected:
                print "    %s" % f_name
    
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    
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
            'kbest__k' : [3, 4, 5, 7, 10, 15, 'all'],
            'svm__kernel' : ['rbf', 'linear'],
            'svm__gamma' : [1e-2, 1e-1, 1, 10],
            'svm__C' : [.1, 1, 10, 100]
        }
        gscv = StratifiedShuffleSplit(labels, n_iter = 500, random_state = 1337)
        clf = GridSearchCV(clf, search_params, scoring='precision', cv=gscv, verbose=1)
        clf.fit(features, labels)
        clf = clf.best_estimator_
    
    if False: # SVM -> Adaboost
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        
        ada = AdaBoostClassifier(SVC(probability=True, kernel='linear'))
        pipeline_steps = [
            ('scaler', MaxAbsScaler()),
            ('kbest', SelectKBest(f_classif)),
            ('ada', ada),
        ]
        clf = Pipeline(pipeline_steps)
    
    if False: # SVM -> Adaboost
        from sklearn.svm import SVC
        from sklearn.ensemble import AdaBoostClassifier
        
        svc = SVC()
        search_params = {
            'kernel' : ['rbf', 'linear'],
            'gamma' : [1e-2, 1e-1, 1, 10],
            'C' : [.1, 1, 10, 100]
        }
        svc_sss = StratifiedShuffleSplit(labels, n_iter = 100, random_state = 1337)
        svm_gscv = GridSearchCV(svc, search_params, scoring='precision', cv=svc_sss, verbose=1)
        svm_gscv.fit(features, labels)
        
        ada = AdaBoostClassifier(
            n_estimators = 200,
            base_estimator = svm_gscv.best_estimator_,
            algorithm = 'SAMME'
        )
        pipeline_steps = [
            ('scaler', MaxAbsScaler()),
            ('kbest', SelectKBest(f_classif)),
            ('ada', ada),
        ]
        clf = Pipeline(pipeline_steps)
#        search_params = {
#            'kbest__k' : [10, 15, 'all'],
#            'ada__n_estimators' : [50, 100, 200, 500]
#        }
#        sss = StratifiedShuffleSplit(labels, n_iter = 500, random_state = 1984)
#        clf = GridSearchCV(pipe, search_params, scoring='recall', cv=sss, n_jobs=5, verbose=1)
        clf.fit(features, labels)
#        clf = clf.best_estimator_
    
    if False: # Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        search_params = {
            'criterion' : ['gini', 'entropy'],
            'splitter' : ['best', 'random'],
            'max_depth' : [None, 5, 10, 20]
        }
        clf = DecisionTreeClassifier()
        gscv = StratifiedShuffleSplit(labels, n_iter = 500, random_state = 1337)
        clf = GridSearchCV(clf, search_params, scoring='precision', cv=gscv, verbose=1)
        clf.fit(features, labels)
        dt = clf.best_estimator_
    
    if False: # AdaBoost
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('kbest', SelectKBest(f_classif)),
#            ('pca', PCA()),
            ('ada', AdaBoostClassifier(algorithm='SAMME'))
        ]
        clf = Pipeline(pipeline_steps)
        
        search_params = {
            'kbest__k' : [3, 4, 5, 7, 10, 15, 'all'],
            'ada__base_estimator' : [
                dt,
#                DecisionTreeClassifier(),
#                DecisionTreeClassifier(max_depth=5),
#                DecisionTreeClassifier(max_depth=10),
            ],
            'ada__n_estimators' : [100, 200, 500]
        }
        gscv = StratifiedShuffleSplit(labels, n_iter = 500, random_state = 1337)
        clf = GridSearchCV(clf, search_params, scoring='recall', cv=gscv, n_jobs=5, verbose=1)
        clf.fit(features, labels)
        clf = clf.best_estimator_
    
    if True: # K Nearest Neighbors
        from sklearn.neighbors import KNeighborsClassifier
        
        pipeline_steps = [
            ('kbest', SelectKBest(f_classif)),
            ('knn', KNeighborsClassifier())
        ]
        clf = Pipeline(pipeline_steps)
        
        search_params = {
            'kbest__k' : [3, 4, 5, 7, 10, 'all'],
            'knn__n_neighbors' : list(range(1,11)),
            'knn__weights' : ['uniform', 'distance'],
            'knn__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'knn__leaf_size' : list(range(5,60,10)),
            'knn__p' : [1,2]
        }
        gscv = StratifiedShuffleSplit(labels, test_size=0.2, random_state=1337)
        clf = GridSearchCV(clf, search_params, scoring='recall', cv=gscv, n_jobs=5, verbose=1)
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
    
    if False: # Stratified K-Fold CV
        #######################################################################
        # K-Fold doesn't work well for this data set because there are only
        # 18 positive samples (POIs), so if you set K higher than 18, some
        # folds won't have any positive samples in them at all... and even
        # with k < 18, there still aren't many positive samples to go off of.
        #######################################################################
        success_f1 = 0.0
        k_fold = StratifiedKFold(labels, 5, shuffle=True, random_state=1337)
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        print "Running Fold Number: ",
        t0 = time.time()
        for k, (train, test) in enumerate(k_fold):
            print k,
            features_train = features[train]
            features_test = features[test]
            labels_train = labels[train]
            labels_test = labels[test]
            
            clf.fit(features_train, labels_train)
            
            accuracies.append(clf.score(features_test, labels_test))
            predictions = clf.predict(features_test)
            precisions.append(precision_score(labels_test, predictions))
            recalls.append(recall_score(labels_test, predictions))
            f1s.append(f1_score(labels_test, predictions))
        print "\n\nK-Fold Cross-Validation took {0:.3f} seconds".format(time.time()-t0)
        acc = np.mean(accuracies)
        prec = np.mean(precisions)
        rec = np.mean(recalls)
        f1 = np.mean(f1s)
        print("Average of {0} Folds -- accuracy: {1:.5f}, precision: {2:.5f}, recall: {3:.5f}, F1 score: {4:.5f}".
              format(k, acc, prec, rec, f1))
    
    if False: # Stratified Shuffle Split CV
        success_f1 = 0.0
        sss = StratifiedShuffleSplit(labels, 100, test_size=0.2, random_state=1337)
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        print "Running Fold Number: ",
        t0 = time.time()
        for k, (train_idx, test_idx) in enumerate(sss):
            print k,
            features_train = []
            features_test  = []
            labels_train   = []
            labels_test    = []
            for ii in train_idx:
                features_train.append( features[ii] )
                labels_train.append( labels[ii] )
            for jj in test_idx:
                features_test.append( features[jj] )
                labels_test.append( labels[jj] )
            
            clf.fit(features_train, labels_train)
            
            accuracies.append(clf.score(features_test, labels_test))
            predictions = clf.predict(features_test)
            precisions.append(precision_score(labels_test, predictions))
            recalls.append(recall_score(labels_test, predictions))
            f1s.append(f1_score(labels_test, predictions))
        print "\n\nStratified Shuffle Split took {0:.3f} seconds".format(time.time()-t0)
        acc = np.mean(accuracies)
        prec = np.mean(precisions)
        rec = np.mean(recalls)
        f1 = np.mean(f1s)
        print("Average of {0} Folds -- accuracy: {1:.5f}, precision: {2:.5f}, recall: {3:.5f}, F1 score: {4:.5f}".
              format(k, acc, prec, rec, f1))
    
    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.
    
    # try-except structure for testing purposes when clf is not defined
    try:
        dump_classifier_and_data(clf, my_dataset, features_list)
    except NameError:
        pass
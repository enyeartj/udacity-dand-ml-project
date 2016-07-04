# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:31:00 2016

@author: John Enyeart
"""

import os
import sys
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

def is_outlier(x, q1, q3, iqr):
    return x < q1 - 1.5*iqr or x > q3 + 1.5*iqr

def make_feature_histograms(dataset, features_list):
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    plt.ioff()
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'hists')):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'hists'))
    for feature, i in zip(features_list[1:], range(len(features[0]))):
        plt.figure()
        feature_values_non_poi = [f[i] for f, l in zip(features, labels) if l == 0.0]
        feature_values_poi = [f[i] for f, l in zip(features, labels) if l == 1.0]
        feature_values = feature_values_non_poi + feature_values_poi
        non_zero_values_non_poi = [x for x in feature_values_non_poi if x != 0.0]
        non_zero_values_poi = [x for x in feature_values_poi if x != 0.0]
        non_zero_values = non_zero_values_non_poi + non_zero_values_poi
        q1, q3 = np.percentile(non_zero_values, [25, 75])
        iqr = q3 - q1
        outliers_hi = [x for x in non_zero_values if is_outlier(x, q1, q3, iqr) and x > q3]
        outliers_lo = [x for x in non_zero_values if is_outlier(x, q1, q3, iqr) and x < q1]
        # get same binwidth for both POI and non-POI
        bins = np.histogram(non_zero_values, bins=50)[1]
        plt.hist(non_zero_values_poi, bins=bins, alpha=.5, lw=0, color='r', label='POIs')
        plt.hist(non_zero_values_non_poi, bins=bins, alpha=.5, lw=0, color='b', label='Non-POIs')
        msg = ('Maximum %s: %d\n' % (feature, max(non_zero_values)) +
               'Minimum %s: %d\n' % (feature, min(non_zero_values)) +
               'Mean %s: %.5f\n' % (feature, np.mean(non_zero_values)) +
               'Median %s: %d\n' % (feature, np.median(non_zero_values)) +
               '\nTotal Number of Values: %d\n' % len(feature_values) +
               'Total Number of Non-Zero Values: %d\n' % len(non_zero_values))
        
        # see which features have low number of non-zero values
        #if float(len(non_zero_values)) / len(feature_values) < 0.5:
        #    print feature
        
        # print out some outlier values if they exist
        for outliers, which_ols in zip([outliers_hi, outliers_lo], ['Top', 'Bottom']):
            if outliers:
                if len(outliers) >= 5:
                    top_n = 5
                else:
                    top_n = len(outliers)
                outliers = sorted(outliers)
                ol_line = q1 - 1.5*iqr
                if which_ols == 'Top':
                    outliers = list(reversed(outliers))
                    ol_line = q3 + 1.5*iqr
                msg += '\n%s %d Outliers: ' % (which_ols, top_n)
                for i in range(top_n):
                    if i != top_n - 1:
                        msg += '%d, ' % outliers[i]
                    else:
                        msg += '%d' % outliers[i]
                plt.axvline(ol_line, lw=.5, ls='--', c='r')
        
        plt.figtext(.3, .4, msg)
        #plt.grid(axis='y')
        plt.title("%s histogram (non-zero values)" % feature)
        plt.legend()
        figname = 'hists/%s_histogram.png' % feature
        plt.savefig(figname)
        plt.close()

def make_feature_scatterplots(df):
    plt.ioff()
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'scatterplots')):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'scatterplots'))
    colors = ['blue', 'red']
    n_features = len(df.columns)
    # assumes first feature (index 0) is 'poi'
    for i in range(1, n_features - 1):
        for j in range(i + 1, n_features):
            plt.figure()
            plt.scatter(df.iloc[:,i], df.iloc[:,j], alpha=.5, c=df.poi.apply(lambda x:colors[int(x)]))
            plt.xlabel(df.columns.values[i])
            plt.ylabel(df.columns.values[j])
            plt.title('%s vs %s scatter plot' % (df.columns.values[j], df.columns.values[i]))
            plt.savefig('scatterplots/%s--%s_scatter.png' % (df.columns.values[j], df.columns.values[i]))
            plt.close()

if __name__ == '__main__':
    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    
    ### The first feature must be "poi".
    features_list = [
        'poi',
        'salary',
        'bonus_per_salary', # new!
        'long_term_incentive_per_salary', # new!
    ]
    for feature in sorted(data_dict['SKILLING JEFFREY K'].keys()):
        if feature != 'email_address' and feature not in features_list:
            features_list.append(feature)
    
    ### Store to my_dataset for easy export below.
    my_dataset = dict()
    # Remove outliers
    remove_me = ['TOTAL']
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
    
    make_feature_histograms(my_dataset, features_list)
    enron_df = featureFormat(my_dataset, features_list, sort_keys=True, as_df=True)
    
    # scatter matrix
    if False:
        plt.ioff()
        plt.figure()
        colors = ['blue', 'red']
        scatter_matrix(enron_df.iloc[:,1:], alpha=0.2, figsize=(50,50), c=enron_df.poi.apply(lambda x:colors[int(x)]))
        plt.savefig('figs/scatter_matrix.png')
    # individual scatter plots
    make_feature_scatterplots(enron_df)
    
    #print enron_df.corr()
    #plt.matshow(enron_df.corr())
    '''
    http://pandas.pydata.org/pandas-docs/stable/visualization.html
    '''
# Machine Learning Project Free-Response Questions

## 1.

> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

The goal of this project is to take data involved in the Enron fraud case and
apply machine learning to that data in order to identify persons of interest
(POIs). Because there are many features that are correlated with being a POI,
machine learning can be used to find which features are the best predictors of
who is a POI, and then use that information to make predictions about other
POIs.

In order to identify outliers, I made a series of histograms to represent POIs
and non-POIs for each feature, and noted which data points were statistical
outliers. The script for generating these histograms can be found in
`visualize_features.py`. This resulted in finding the 'TOTAL' data point, which
should obviously be left out because 'TOTAL' can't be a person of interest. I
also noticed and removed another one that shouldn't be in there: 'THE TRAVEL
AGENCY IN THE PARK'. The other outliers found tended to be POIs such as the CEO,
Jeffrey Skilling, who obviously shouldn't be left out, so no other outliers were
removed. I also left out Eugene Lockhart because there was no data associated
with him, so leaving him in could have skewed the results.

## 2.

> What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

The features that ended up being used in the POI identifier were the ones
selected by SelectKBest in the classifier pipeline. SelectKBest was included in
the GridSearchCV pipeline, so various values of `k` were tested, but the final
classifier ended up with `k = 4`, which SelectKBest chose as the following (with
scores in parentheses):

1. `exercised_stock_options` (24.8151)
2. `total_stock_value` (24.1829)
3. `bonus` (20.7923)
4. `salary` (18.2897)

Aside from these, the next highest score from SelectKBest was only 11.4584.
These features make sense because the POIs in the Enron case were known to have
high bonuses and salaries, but particularly, they tried to sell off their stock
before everything came crashing down, so it is no wonder that
`exercised_stock_options` came out with the highest score.

Scaling makes a difference for selecting the k nearest neighbors of a data
point, so I tried using sklearn's StandardScaler, but I found that my classifier
performed better without scaling, so it was left out in the final version.

I attempted to make several features:

1. `bonus_per_salary`: This is a person's bonus divided by their salary. The
reasoning behind this metric is that POIs would probably have a larger
bonus-to-salary ratio.
2. `long_term_incentive_per_salary`: This is a person's long term incentive
divided by their salary. The reasoning is the same as `bonus_per_salary`.
3. `important_to`: I parsed the emails to find which words were most important
in regards to selecting a POI. This was done in `get_words_from_emails.py`.
`important_to` is the number of important words in this person's emails that
were sent to POIs.
4. `important_from`: Similar to `important_to`, `important_from` is the number
of important words this person received from POIs.

Ultimately, I didn't end up using any of these metrics because SelectKBest did
not pick any of them.

## 3.

> What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

I ended up using the KNearestNeighbors algorithm. I also tried using
Gaussian Naive Bayes, SVM, SVM with Adaboost, and Decision Tree
with Adaboost as classifiers. Performance for each is listed here:

- `KNearestNeighbors`: Accuracy: 0.85753	Precision: 0.45240	Recall: 0.32550	F1: 0.37860	F2: 0.34485
- `KNearestNeighbors` (with scaling): Accuracy: 0.83127	Precision: 0.35764	Recall: 0.33350	F1: 0.34515	F2: 0.33806
- `GaussianNB`: Accuracy: 0.72047	Precision: 0.22946	Recall: 0.46500	F1: 0.30729	F2: 0.38580
- `SVC`: Accuracy: 0.85947	Precision: 0.42703	Recall: 0.15800	F1: 0.23066	F2: 0.18078
- `SVC` with `Adaboost`: Accuracy: 0.86673	Precision: 0.55556	Recall: 0.00250	F1: 0.00498	F2: 0.00312
- `DecisionTree` with `Adaboost`: Accuracy: 0.83427	Precision: 0.35449	Recall: 0.29600	F1: 0.32262	F2: 0.30610

## 4.

> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).

Tuning the parameters of an algorithm affects how the algorithm performs and
occurs in the creation of the algorithm before fitting it to any data. For
example, with the KNearestNeighbors algorithm, you can tell the algorithm how
many neighbors to look at when making a decision with the `k_nearest` parameter,
tell it whether to weight the nearest neighbors uniformly or based on distance
with the `weights` parameter, etc. All of these parameters will affect how the
algorithm makes its decisions, and getting parameters tuned to an optimal state
is essential for good performance. If parameters are off, several problems can
occur from general low performance to overfitting.

In this project, I used sklearn's `GridSearchCV` to tune the parameters of the
algorithm. `GridSearchCV` lets you input a list of different paramter values to
try for whichever parameters you specify, and then tests all different
combinations of the specified parameters. It scores the performance for each
combination and then lets you know which combination performed the best.

## 5.

> What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

Validation is a way of validating the performance of a classifier. After the
classifier is fit to a dataset, an independent dataset that was was not used in
fitting is used to guage the performance of the classifier, and thus "validate"
it. This also serves as a check on overfitting because if a classifier has been
overfit to a dataset, it will perform very well on the data it was fit to, but
poorly on data it hasn't seen. This is called overfitting because the classifier
essentially just memorized all the right answers and therefore will not produce
a correct answer. A classic mistake is to split training and validation sets in
an unbalanced way. For example, in this project, there are only 18 POIs out of
144 people. So, if you train on a set that contains all of the POIs and validate
on a set that has none, you won't be able to see how well your algorithm makes
positive identifications. Likewise, if you train on data that has no POIs and
validate on data that has all of them, the training will essentially tell the
classifier that you will be 100% accurate if you just say nobody is a POI.
Therefore, it is important to split training and validation sets in a way that
has a balance of POIs and non-POIs across both sets.

I ended up using `StratifiedShuffleSplit` to validate my analysis.
`StratifiedShuffleSplit` randomizes the datasets and then splits a certain
percentage off to be used for training and the rest to be used for validation
while making sure that the percentage of POIs in each set stay constant.

## 6.

> Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

The main evaluation metrics that were used in this analysis were precision and
recall. Precision describes the percentage of positive identifications that
were correct. The algorithm's precision score of 0.45 means that 45% of the
people that it identified as POIs were actually POIs. Recall describes the
percentage of all positive samples that were classified as positive. The
algorithm's score of 0.33 means that 33% of POIs were correctly identified.
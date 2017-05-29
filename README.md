# Machine Learning Final Project

### Udacity Data Analyst Nanodegree

This is the final project for the machine learning course in Udacity's Data
Analyst Nanodegree. The goal of this project was to take data regarding the
Enron fraud case, train a machine learning algorithm on that data, and then use
the algorithm to classify whether an individual was likely to be a person of
interest (POI) or not.

As described in the project overview from Udacity:

> In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

In the project, I gathered features from the data and created a few new features
to be used in training the machine learning algorithm. Questions and answers
regarding the algorithm, feature selection, evaluation metrics, etc. can be
found in `project_questions.md`.

----

- Features were explored using `visualize_features.py` and formatted using
`feature_format.py`.
- The main training and algorithm selection was performed in the file `poi_id.py`.
- Custom email features were generated with `get_words_from_emails.py` and
`parse_out_email_text.py`.
- The algorithm was tested using `tester.py`.

----

## Sources and References

- Udacity's Data Analyst Nanodegree [Intro to Machine Learning Course](https://www.udacity.com/course/intro-to-machine-learning--ud120)
- Scikit-learn's [documentation](http://scikit-learn.org/stable/user_guide.html)
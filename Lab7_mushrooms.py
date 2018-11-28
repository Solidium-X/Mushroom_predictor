import pandas
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import sklearn.metrics as metrics
import numpy as np


# In this lab, we work with a mushroom dataset.
# https://www.kaggle.com/uciml/mushroom-classification

# We'll build a simple classifier to predict whether mushrooms are poisonous!

# Our test dataset has a 'class' column (p = poisonous, e = edible), and other feature columns.
# We're going to train a classifier to predict this class (p vs e) based on the other features.
# There are two classes, so this is a binary classifer.
# We'll use a very simple Naive Bayes Classifier - it picks the class with the highest probability
# given the input features.


# +-------------------+----------+
# | EdibleOrPoisonous | RedColor |
# +-------------------+----------+
# | E                 | Y        |
# | E                 | Y        |
# | E                 | Y        |
# | P                 | N        |
# | P                 | Y        |
# | P                 | N        |
# +-------------------+----------+
#
# Based on these examples, if we see a red mushroom - is it more likely to be poisonous or edible?
#
# We apply Bayes' Rule:
# P(A|B) = P(B|A) * P(A) / P(B)
#
# We can estimate the probabilities by looking at the mushrooms we've seen so far (the table above).
# We can ignore the denominators, because they are the same for both cases).
#
# P(Edible|Red) = P(Red|Edible) * P(Edible) / P(Red)
#               = 3/3 * 3/6 / 4/6
#               = 1 * 1/2 / 4/6
#               = 3/4
# P(Poisonous|Red) = P(Red|Poisonous) * P(Poisonous) / P(Red)
#                  = 1/3 * 3/6 / 4/6
#                  = 1/4
# So,
#  P(Edible|Red) > P(Poisonous|Red) -- its more likely edible than poisonous -- so lets eat the mushroom!


# This example uses one feature 'redcolor', we can extend this to more than one feature by multiplying their probabilities:
# https://www.khanacademy.org/math/ap-statistics/probability-ap/probability-multiplication-rule/a/general-multiplication-rule

# Its called 'naive' because we make the 'naive' modelling assumption that the features are statistically independent from eachother, so that we can multiply them in a simple way.

# Also, in practice, we apply 'smoothing' - assuming a few other mushrooms exist with all combinations of features.

# Bayesian classifiers are still used as a baseline, particularly in NLP problems.
# The features need to be binary probabilistic events - so we need to convert any categorical data
# to binary features.

# We'll partition our input dataset into two: training and testing. For a 'fair' evaluation we classify
# only mushrooms in the test portion - that the classifier hasn't seen before!

# We'll use pandas - a library similar to numpy, for working with datasets, and we'll use scikit-learn for the machine learning.

TEST_TRAIN_SPLIT = 5


all_mushrooms = pandas.read_csv('Lab7_mushrooms.csv')
# Here's the top two lines of the file for reference...
# class,cap-shape,cap-surface,cap-color,bruises,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring,stalk-surface-below-ring,stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat
# p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u


all_mushrooms = all_mushrooms.drop_duplicates()

classes_all = all_mushrooms.iloc[:, 0]
# Convert the classes to integers:
classes_all = classes_all.map(lambda pos_or_edible: 1 if pos_or_edible == 'p' else 0)

print('\nHow balanced are the classes?')
# TODO: count the number of elements in classes_all with class 0 or 1 respectively. Which is which?
print(f'Poisonous Mushrooms: {len(classes_all.loc[classes_all == 1])}')
print(f'Edible Mushrooms: {len(classes_all.loc[classes_all == 0])}')


# Remove the labels to leave just the features (all the columns but the first):
features_all = all_mushrooms.iloc[:, 1:]

# These values are categorical, to use a Naive Bayes classifier, they need to be binary features:
features_all = pandas.get_dummies(features_all)

# Now we split training and test data...
# TODO: complete this code, so that every 5th row is selected as test data, and all the others are training data.
# Hint: use the index column, take a modulo, and compare with TEST_TRAIN_SPLIT
# Hint: take a look at the cheat sheet - especially the example marked as 'Select rows meeting logical condition, and only the specific columns'
test_classes_golden = classes_all[classes_all.index % 5 == 0]
training_classes = classes_all[classes_all.index % 5 != 0]
test_features = features_all[features_all.index % 5 == 0]
training_features = features_all[features_all.index % 5 != 0]
#print(test_classes_golden)
#print(test_features)

# Lets check we split things up correctly...
assert (len(test_classes_golden) == len(test_features))
assert (len(training_classes) == len(training_features))
print('\ntraining/test/total split:')
print(len(training_classes), len(test_classes_golden), len(all_mushrooms))


# Task 1: Train a Naive Bayesian Classifier, on the 'training' mushrooms, and predict the class of the test mushrooms (i.e. can we eat them!):
# Hint: use the sklearn.naive_bayes.BernoulliNB class - this is the simplest model, suitable for binary features.
# Specify and alpha value of 1 (Laplace smoothing), don't specify the other parameters.

# Bernoulli attempt
bernNB = BernoulliNB()
bernNB.fit(training_features, training_classes)
print('\n', bernNB)

expect = test_classes_golden
bern_prediction = bernNB.predict(test_features)
print('Accuracy = ', metrics.accuracy_score(expect, bern_prediction))
print('F1 score = ', metrics.f1_score(expect, bern_prediction))

# Multinomia attempt
multiNB = MultinomialNB()
multiNB.fit(training_features, training_classes)
print('\n', multiNB)

multi_prediction = multiNB.predict(test_features)
print('Accuracy = ', metrics.accuracy_score(expect, multi_prediction))
print('F1 score = ', metrics.f1_score(expect, multi_prediction))

# Task 2: Compare your predictions to the true class values for those mushrooms. Compute the number of True Positives, True Negatives, False Positives, and False Negatives, and print them.
# Hint: create a DataFrame from the numpy array returned by the classifier. Specify a name for the column, e.g. 'prediction'
# Merge this with the real test classes, joining left and right indices.
# Gotcha: Boolean operators like 'and' are considered ambiguous for Pandas, you can use bitwise operator like '&'
# See: https://stackoverflow.com/questions/36921951 or use np.logical_and(..)



# Task 3: Compute the F1 score, as explained here: https://en.wikipedia.org/wiki/F1_score

# Extension:

# Try to implement your own Naive Bayes classifier class from scratch - and compare your results.
# Assume binary features. Don't forget the smoothing!
# Its not as hard as you might think...
# https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_naive_Bayes

import copy
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from classifier import load_k_fold_data, getFeatures, getLabels
from hw3_utils import load_data
#
from our_hw3_utils import sbs, utilityFunction, sfs

test_group_features, test_group_labels = load_k_fold_data(2)
train_group = load_k_fold_data(1)
data = getFeatures(train_group)
labels = getLabels(train_group)
train_features, train_labels, test_features = load_data()

#
# with open("experiments/test.csv", "w") as csv_file:
#     writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
#     for row in test_features:
#         writer.writerow(row)
# #
# with open("experiments/tag.csv", "w") as csv_file:
#     writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
#     for row in labels:
#         writer.writerow([row])


# X_train = data
# y_train = getLabels(train_group)
# X_test = test_group_features
# y_test = test_group_labels

# TODO: DecisionTree
# clf = DecisionTreeClassifier()
# clf.fit(data, getLabels(train_group))
# y = clf.predict(test_group_features)
# print(accuracy_score(y, test_group_labels))
# print(clf.score(test_group_features, test_group_labels))

# print(
#     'Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(data, getLabels(train_group))))
# print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(
#     clf.score(test_group_features, test_group_labels)))
#
# # TODO: ID3
# clf = DecisionTreeClassifier(criterion="entropy").fit(data, getLabels(train_group))
# print(
#     'Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(data, getLabels(train_group))))
# print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(
#     clf.score(test_group_features, test_group_labels)))

factor = {
    "ID3": 1,
    "3-Nearest Neighbors": 4,
    "Neural Net": 4,
    "AdaBoost": 3,
    "RandomForest": 3,
}


# names = ["ID3", "3-Nearest Neighbors", "Neural Net", "AdaBoost", "RandomForest"]
names = ["ID3", "MinLeaf", "MinLeaf2", "1-Nearest Neighbors", "2-Nearest Neighbors", "3-Nearest Neighbors",
         "RandomForest"]



# DecisionTreeClassifier(criterion="entropy", max_depth= 2, min_samples_leaf= 20)
classifiers = [
    DecisionTreeClassifier(criterion="entropy"),
    DecisionTreeClassifier(min_samples_leaf = 3),
    KNeighborsClassifier(1),
    KNeighborsClassifier(1),
    KNeighborsClassifier(1),
    KNeighborsClassifier(2),
    KNeighborsClassifier(3),
    RandomForestClassifier()]

X_train = data
y_train = getLabels(train_group)
X_test = test_group_features
y_test = test_group_labels


def convertFeaturesToResFet(X_train, y_train, features):
    clf_results = []
    for name, clf in zip(names, classifiers):
        # clf_factor = factor[name]
        clf.fit(X_train, y_train)
        train_predict = clf.predict(features)

        if len(clf_results) == 0:
            # first classifier create nodes
            for prediction in train_predict:
                clf_results.append([prediction])
        else:
            # non-first classifier append to nodes
            for prediction, l in zip(train_predict, clf_results):
                l.append(prediction)
    return clf_results


def newClassifier(X_train, y_train, X_test, y_test):
    clf_results_train = convertFeaturesToResFet(X_train, y_train, X_train)
    clf_results_test = convertFeaturesToResFet(X_train, y_train, X_test)

    new_train = []
    for features, extra in zip(X_train, clf_results_train):
        extra = np.array(extra)
        new_train.append(np.append(features, extra))

    new_test = []
    for features, extra in zip(X_test, clf_results_test):
        extra = np.array(extra)
        new_test.append(np.append(features, extra))

    # clf_results_train = X_train.extend(clf_results_train)
    # clf_results_test = X_test.extend(clf_results_test)

    # at this point clf_factor contains 499 samples of features: each feature is the result of a certain classifier and now we'll train a new classifer using this features
    final_clf = classifiers[2]
    final_clf.fit(new_train, y_train)

    score_train = final_clf.score(new_train, y_train)
    score_test = final_clf.score(new_test, y_test)
    # print('Accuracy of ' + "MAXX" + ' on training set: ' + str(score_train))
    # print('Accuracy of ' + "MAXX" + ' on test set: ' + str(score_test))
    return score_train, score_test


# newClassifier()
# test_avg = 0
# train_avg = 0
# for i in range(5):
#     print(i)
#     score_train, score_test = newClassifier()
#     train_avg += score_train
#     test_avg += score_test
#     print('current accuracy of ' + "MAXX" + ' on test set: ' + str(score_test))
#
# print('Avg accuracy of ' + "MAXX" + ' on training set: ' + str(train_avg/50))
# print('Avg accuracy of ' + "MAXX" + ' on test set: ' + str(test_avg/50))

# TODO: feature selection
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X_train, y_train)
# score = neigh.score(X_test, y_test)
# features_chosen = sfs(X_train, y_train, 10, neigh, utilityFunction)
# # create x_train_subset and x_test_subset containing only values of chosen features
# X_train_subset = [[] for i in range(len(X_train))]
# X_test_subset = [[] for i in range(len(X_test))]
# for feature_index in features_chosen:
#     for object_index in range(len(X_test)):
#         X_test_subset[object_index].append(X_test[object_index][feature_index])
#     for object_index in range(len(X_train)):
#         X_train_subset[object_index].append(X_train[object_index][feature_index])
path = "updated_features.data"
# with open(path, 'wb') as f:
#     tuple_to_store = X_train_subset, X_test_subset
#     pickle.dump(tuple_to_store, f, protocol=pickle.HIGHEST_PROTOCOL)

# x_train_subset and x_test_subset are updated with relevant features only

# newClassifier(X_train_subset, y_train, X_test_subset, y_test)

with open(path, 'rb') as f:
    X_train_subset, X_test_subset = pickle.load(f)

test_avg = 0
train_avg = 0
iterations = 50
for i in range(iterations):
    print(i)
    score_train, score_test = newClassifier(X_train_subset, y_train, X_test_subset, y_test)
    train_avg += score_train
    test_avg += score_test
    print('current accuracy of ' + "MAXX" + ' on test set: ' + str(score_test))

print('Avg accuracy of ' + "MAXX" + ' on training set: ' + str(train_avg / iterations))
print('Avg accuracy of ' + "MAXX" + ' on test set: ' + str(test_avg / iterations))

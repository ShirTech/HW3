import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from hw3_utils import load_data
from our_hw3_utils import feature_selection

factor = {
    "ID3": 1,
    "3-Nearest Neighbors": 4,
    "Neural Net": 4,
    "AdaBoost": 3,
    "RandomForest": 3,
}

# names = ["ID3", "3-Nearest Neighbors", "Neural Net", "AdaBoost", "RandomForest"]
names = ["ID3", "MinLeaf", "vd", "MinLeafedf", "MinLeaf2", "1-Nearest Neighbors", "2-Nearest Neighbors",
         "3-Nearest Neighbors", "VD", "RandomForest"]

# DecisionTreeClassifier(criterion="entropy", max_depth= 2, min_samples_leaf= 20)
classifiers = [
    DecisionTreeClassifier(criterion="entropy"),
    DecisionTreeClassifier(min_samples_leaf=3),
    DecisionTreeClassifier(criterion="entropy", min_samples_leaf=19, max_depth=3),
    KNeighborsClassifier(1),
    KNeighborsClassifier(1),
    KNeighborsClassifier(1),
    KNeighborsClassifier(2),
    KNeighborsClassifier(2),
    KNeighborsClassifier(3),
    RandomForestClassifier()]


# X_train = data
# y_train = getLabels(train_group)
# X_test = test_group_features
# y_test = test_group_labels


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

#
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

# x_train_subset and x_test_subset are updated with relevant features only

# newClassifier(X_train_subset, y_train, X_test_subset, y_test)
#
# with open("updated_features.data", 'rb') as f:
#     X_train_subset, X_test_subset = pickle.load(f)
#
# test_avg = 0
# train_avg = 0
# iterations = 50
# for i in range(iterations):
#     print(i)
#     score_train, score_test = newClassifier(X_train_subset, y_train, X_test_subset, y_test)
#     train_avg += score_train
#     test_avg += score_test
#     print('current accuracy of ' + "MAXX" + ' on test set: ' + str(score_test))
#
# print('Avg accuracy of ' + "MAXX" + ' on training set: ' + str(train_avg / iterations))
# print('Avg accuracy of ' + "MAXX" + ' on test set: ' + str(test_avg / iterations))
#

def test():
    train_features, train_labels, test_features = load_data()
    X_train_subset = feature_selection(train_features, train_labels)
    # with open("updated_features_2.data", 'rb') as f:
    #     X_train_subset = pickle.load(f)

    total_avg = 0
    for k in [2, 4, 6, 8, 10]:
        test_size = 1 / k
        total_score_per_k_train = 0
        total_score_per_k_test = 0
        for i in range(k):
            X_train, X_test, y_train, y_test = train_test_split(X_train_subset, train_labels, test_size=test_size,
                                                                random_state=i)
            clf = MAXX(X_train, y_train)
            total_score_per_k_train += clf.test(X_train, y_train)
            total_score_per_k_test += clf.test(X_test, y_test)
        print('K value is: ' + str(k))
        print('Avg accuracy of MAXX on test set: ' + str(total_score_per_k_test / k))
        total_avg += total_score_per_k_test / k
    print('Total avg for all k is: ' + str(total_avg / 5))


class MAXX():
    def __init__(self, data, labels):
        self.X = data
        self.y = labels

    def classify(self, features):
        clf_results_features = convertFeaturesToResFet(self.X, self.y, features)
        new_features = []
        for features, extra in zip(features, clf_results_features):
            extra = np.array(extra)
            new_features.append(np.append(features, extra))

        return self.clf.predict(new_features)

    def test(self, X_test, y_test):
        X_train = self.X
        y_train = self.y
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

        final_clf = KNeighborsClassifier(1)
        final_clf.fit(new_train, y_train)

        score_test = final_clf.score(new_test, y_test)
        return score_test


test()

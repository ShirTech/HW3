import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from hw3_utils import load_data, write_prediction
from our_hw3_utils import feature_selection

names = ["ID3", "MinLeaf", "limited_ID3", "1-Nearest Neighbors", "1-Nearest Neighbors", "1-Nearest Neighbors",
         "2-Nearest Neighbors", "2-Nearest Neighbors", "3-Nearest Neighbors", "RandomForest"]

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


def convertFeaturesToResultFeatures(X_train, y_train, features):
    clf_results = []
    for name, clf in zip(names, classifiers):
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


def test():
    train_features, train_labels, test_features = load_data()
    X_train_subset, X_test_subset = feature_selection(train_features, train_labels)

    total_avg = 0
    for k in [2, 4, 6, 8, 10]:
        test_size = 1 / k
        total_score_per_k_train = 0
        total_score_per_k_test = 0
        for i in range(k):
            X_train, X_test, y_train, y_test = train_test_split(X_train_subset, train_labels, test_size=test_size,
                                                                random_state=i)
            clf = CompetitionClassifier(X_train, y_train)
            total_score_per_k_train += clf.test(X_train, y_train)
            total_score_per_k_test += clf.test(X_test, y_test)
        print('K value is: ' + str(k))
        print('Avg accuracy of MAXX on test set: ' + str(total_score_per_k_test / k))
        total_avg += total_score_per_k_test / k
    print('Total avg for all k is: ' + str(total_avg / 5))

    pred = clf.classify(X_test_subset)
    write_prediction(pred)


class CompetitionClassifier():
    def __init__(self, data, labels):
        self.X = data
        self.y = labels

    def classify(self, X_test):
        X_train = self.X
        y_train = self.y
        clf_results_train = convertFeaturesToResultFeatures(X_train, y_train, X_train)
        clf_results_test = convertFeaturesToResultFeatures(X_train, y_train, X_test)

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

        prediction = final_clf.predict(X_test)
        return prediction

    def test(self, X_test, y_test):
        X_train = self.X
        y_train = self.y
        clf_results_train = convertFeaturesToResultFeatures(X_train, y_train, X_train)
        clf_results_test = convertFeaturesToResultFeatures(X_train, y_train, X_test)

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

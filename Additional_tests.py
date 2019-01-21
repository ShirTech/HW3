import csv

from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

from classifier import evaluate
from hw3_utils import abstract_classifier, abstract_classifier_factory


class id3_classifier(abstract_classifier):
    def __init__(self, clf):
        self.clf = clf
    '''
    classify a new set of features
    :param features: the list of feature to classify
    :return: a tagging of the given features (1 or 0)
    '''

    def classify(self, features):
        return self.clf.predict([features])


class id3_factory(abstract_classifier_factory):
    '''
    train a classifier
    :param data: a list of lists that represents the features that the classifier will be trained with
    :param labels: a list that represents  the labels that the classifier will be trained with
    :return: id3 classifier object
    '''
    def train(self, data, labels):
        clf = DecisionTreeClassifier(criterion="entropy").fit(data, labels)
        return id3_classifier(clf)


class perceptron_classifier(abstract_classifier):
    def __init__(self, clf):
        self.clf = clf
    '''
    classify a new set of features
    :param features: the list of feature to classify
    :return: a tagging of the given features (1 or 0)
    '''

    def classify(self, features):
        return self.clf.predict([features])


class perceptron_factory(abstract_classifier_factory):
    '''
    train a classifier
    :param data: a list of lists that represents the features that the classifier will be trained with
    :param labels: a list that represents  the labels that the classifier will be trained with
    :return: id3 classifier object
    '''
    def train(self, data, labels):
        return perceptron_classifier(Perceptron(max_iter=5, tol=None).fit(data, labels))


def Additional_tests():
    results = []

    id3 = id3_factory()
    avg_accuracy, avg_error = evaluate(id3, 2)
    results.append([1, avg_accuracy, avg_error])

    perceptron = perceptron_factory()
    avg_accuracy, avg_error = evaluate(perceptron, 2)
    results.append([2, avg_accuracy, avg_error])

    with open("experiments12.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        for row in results:
            writer.writerow(row)

import csv
import math
import pickle
import random
from pathlib import Path
import heapq

from hw3_utils import abstract_classifier, load_data, abstract_classifier_factory
from collections import Counter

train_features, train_labels, test_features = load_data()


def majority(arr):
    classification_dictionary = Counter(arr)
    for (key, val) in classification_dictionary.items():
        if val > (len(arr) / 2):
            return key


def distance_euclidean(list1, list2) -> float:
    sum = 0
    for x, y in zip(list1, list2):
        sum += (x - y) ** 2
    return math.sqrt(sum)


class knn_classifier(abstract_classifier):

    def __init__(self, k, train_features, train_labels):
        self.k = k
        self.data = train_features
        self.labels = train_labels

    '''
    classify a new set of features
    :param features: the list of feature to classify
    :return: a tagging of the given features (1 or 0)
    '''

    def classify(self, features):
        # TODO 1. calculate distance from examples using distance_euclidean
        distances = []
        for testee in self.data:
            current_distance = distance_euclidean(features, testee)
            distances.append(current_distance)

        #  TODO 2. find k nearest examples
        results = []
        labels = self.labels[0]
        distances, classifications = (list(t) for t in zip(*sorted(zip(distances, labels))))
        k = min(self.k, len(labels))
        for i in range(k):
            results.append(classifications[i])

        #  TODO 3. choose majority from those examples
        return majority(results)


class knn_factory(abstract_classifier_factory):
    def __init__(self, k):
        self.k = k

    '''
    train a classifier
    :param data: a list of lists that represents the features that the classifier will be trained with
    :param labels: a list that represents  the labels that the classifier will be trained with
    :return: abstruct_classifier object
    '''

    def train(self, data, labels) -> knn_classifier:
        return knn_classifier(self.k, data, labels)


def split_crosscheck_groups(dataset, num_folds):
    healthy_list = []
    sick_list = []
    features_list = dataset[0]
    labels = dataset[1]
    for features, label in zip(features_list, labels):
        if label == True:
            sick_list.append((features, label))
        else:
            healthy_list.append((features, label))

    fold_size = len(labels) / num_folds
    healthy_len = len(healthy_list)
    sick_len = len(sick_list)
    healthy_in_fold = math.floor((healthy_len / (healthy_len + sick_len)) * fold_size)
    sick_in_fold = math.floor((sick_len / (healthy_len + sick_len)) * fold_size)
    shuffled_healthy = sorted(healthy_list, key=lambda L: random.random())
    shuffled_sick = sorted(sick_list, key=lambda L: random.random())

    for i in range(1, num_folds + 1):
        temp_features = []
        temp_labels = []
        # file configurations:
        path = "ecg_fold_" + str(i) + ".data"
        for j in range(healthy_in_fold):
            tmp = shuffled_healthy.pop()
            temp_features.append(tmp[0])
            temp_labels.append(tmp[1])

        for j in range(sick_in_fold):
            tmp = shuffled_sick.pop()
            temp_features.append(tmp[0])
            temp_labels.append(tmp[1])

        # if i == num_folds:
        #     if shuffled_sick

        with open(path, 'wb') as f:
            tuple_to_store = (temp_features, temp_labels)
            pickle.dump(tuple_to_store, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_k_fold_data(index):
    path = "ecg_fold_" + str(index) + ".data"
    with open(path, 'rb') as f:
        features, labels = pickle.load(f)
    return features, labels


def evaluate(classifier_factory, k):
    accuracy_sum = 0
    error_sum = 0
    for i in range(1, k + 1):
        # TODO: choose i to be the test group and unite the rest for training
        test_group_features, test_group_labels = load_k_fold_data(i)
        train_group = groups_union(i, k)

        # TODO: train classifier with train data
        data = getFeatures(train_group)[0]
        classifier = classifier_factory.train(data, getLabels(train_group))

        # TODO: classify each item in test group and collect statistics
        success = 0
        failure = 0
        for testee_features, testee_real_label in zip(test_group_features, test_group_labels):
            given_label = classifier.classify(testee_features)
            if given_label == testee_real_label:
                success += 1
            else:
                failure += 1
        accuracy = success / (success + failure)
        error = 1 - accuracy
        accuracy_sum += accuracy
        error_sum += error

    avg_accuracy = accuracy_sum / k
    avg_error = error_sum / k
    return avg_accuracy, avg_error


def groups_union(index, k):
    features = []
    labels = []
    for i in range(1, k + 1):
        if i != index:
            train_group_features, train_group_labels = load_k_fold_data(i)
            features.append(train_group_features)
            labels.append(train_group_labels)
    return features, labels


def getFeatures(train_group):
    return train_group[0]


def getLabels(train_group):
    return train_group[1]


x = (train_features, train_labels)
split_crosscheck_groups(x, 2)

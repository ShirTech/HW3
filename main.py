import math

from Additional_tests import Additional_tests
from CompeteitionClassifier import compete
from KNN_test import KNN_test
from classifier import split_crosscheck_groups
from hw3_utils import load_data


def main():
    train_features, train_labels, test_features = load_data()
    x = (train_features, train_labels)
    # split_crosscheck_groups(x, 2)

    KNN_test()
    Additional_tests()
    # compete()


main()

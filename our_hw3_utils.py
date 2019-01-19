import numpy as np
from sklearn.model_selection import cross_val_score
import copy


def sfs(x, y, required_features_num, clf, score):
    number_of_features = len(x[0])
    features_chosen = []
    # x subset will contain required features only
    x_subset = [[] for i in range(len(x))]

    while len(features_chosen) < required_features_num:
        best_score = -1
        best_index = -1

        for feature_index in range(number_of_features):
            if feature_index in features_chosen:
                continue

            # create x_subset_temp containing values for all chosen features and the new feature the we are considering
            x_subset_temp = copy.deepcopy(x_subset)
            for feature_list in range(len(x)):
                x_subset_temp[feature_list].append(x[feature_list][feature_index])

            current_score = score(clf, x_subset_temp, y)
            if current_score > best_score:
                best_score = current_score
                best_index = feature_index

                # update features_chosen to contain the new feature's index and x_subset to contain its values
        features_chosen.append(best_index)
        for feature_list in range(len(x)):
            x_subset[feature_list].append(x[feature_list][best_index])

    return features_chosen


def sbs(x, y, k, clf, score):
    """
    :param x: feature set to be trained using clf. list of lists.
    :param y: labels corresponding to x. list.
    :param k: number of features to select. int
    :param clf: classifier to be trained on the feature subset.
    :param score: utility function for the algorithm, that receives clf, feature subset and labeles, returns a score.
    :return: list of chosen feature indexes
    """

    import copy
    number_of_features = len(x[0])

    # indecis of chosen features
    features_for_removal = []

    # x containg only values of chosen features
    x_subset = copy.deepcopy(x)

    while len(features_for_removal) < k:
        min_score = 2
        best_index = -1

        for feature_index in range(number_of_features):
            # feature already removed, don't consider it anymore
            if feature_index in features_for_removal:
                continue

            # create x_subset_temp containing values for all chosen features and the new feature the we are considering
            x_subset_temp = copy.deepcopy(x_subset)
            for object_index in range(len(x)):
                x_subset_temp[object_index][feature_index] = 0

            temp_score = score(clf, x_subset_temp, y)
            if temp_score < min_score:
                min_score = temp_score
                best_index = feature_index

        # update features_for removal to contain the new feature's index and x_subset to contain its values
        features_for_removal.append(best_index)
        for object_index in range(len(x)):
            x_subset[object_index][best_index] = 0

    return features_for_removal


def utilityFunction(clf, x, y):
    # estimate features subset using 4 cross validation
    # to be used as score function in sfs algorithm
    scores = cross_val_score(clf, x, y, cv=4)
    return scores.mean()

#
# features_for_removal = sbs(X_train, y_train, 4, neigh, utilityFunction)
#
# # create x_train_subset and x_test_subset containing only values of chosen features
# x_train_subset = copy.deepcopy(X_train)
# x_test_subset = copy.deepcopy(X_test)
# for feature_index in features_for_removal:
#     for object_index in range(len(X_test)):
#         x_test_subset[object_index][feature_index] = 0
#         # x_test_subset[object_index] = np.delete(x_test_subset[object_index], feature_index)
#     for object_index in range(len(X_train)):
#         x_train_subset[object_index][feature_index] = 0
#         # x_train_subset[object_index] = np.delete(x_train_subset[object_index], feature_index)
#
#
# neigh.fit(x_train_subset, y_train)
# score2 = neigh.score(x_test_subset, y_test)
# print(score2)

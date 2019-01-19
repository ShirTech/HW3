import copy

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def sfs(x, y, required_features_num, clf, score):
    number_of_features = len(x[0])
    number_of_samples = len(y)
    chosen_features = []

    # TODO: x subset will contain required features only
    x_subset = [[] for i in range(len(x))]

    for i in range(required_features_num):
        best_score = -1
        best_index = -1

        for feature_index in range(number_of_features):

            if feature_index not in chosen_features:

                x_subset_temp = copy.deepcopy(x_subset)
                for feature_list in range(number_of_samples):
                    x_subset_temp[feature_list].append(x[feature_list][feature_index])

                current_score = score(clf, x_subset_temp, y)
                if current_score > best_score:
                    best_score = current_score
                    best_index = feature_index

        # TODO: update chosen_features to contain the new feature's index and x_subset to contain its values
        chosen_features.append(best_index)
        for feature_list in range(len(x)):
            x_subset[feature_list].append(x[feature_list][best_index])

    return chosen_features


def evaluate_features(clf, x, y):
    scores = cross_val_score(clf, x, y, cv=4)
    return scores.mean()


def feature_selection(X, y, X_test):
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(X, y)

    features_chosen = sfs(X, y, 10, KNN, evaluate_features)

    X_train_subset = [[] for i in range(len(X))]
    X_test_subset = [[] for i in range(len(X_test))]
    for feature_index in features_chosen:
        for object_index in range(len(X_test)):
            X_test_subset[object_index].append(X_test[object_index][feature_index])
        for object_index in range(len(X)):
            X_train_subset[object_index].append(X[object_index][feature_index])

    return X_train_subset, X_test_subset

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from classifier import load_k_fold_data, getFeatures, getLabels
from hw3_utils import load_data

#
test_group_features, test_group_labels = load_k_fold_data(1)
train_group = load_k_fold_data(2)
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
#
# # TODO: GINI
# clf = DecisionTreeClassifier(criterion="gini").fit(data, getLabels(train_group))
# print(
#     'Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(data, getLabels(train_group))))
# print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(
#     clf.score(test_group_features, test_group_labels)))

# # TODO: GaussianNB
# clf = GaussianNB().fit(data, getLabels(train_group))
# print(
#     'Accuracy of GaussianNB classifier on training set: {:.2f}'.format(clf.score(data, getLabels(train_group))))
# print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(
#     clf.score(test_group_features, test_group_labels)))
#
# # TODO: KNN
# clf = KNeighborsClassifier(3).fit(data, getLabels(train_group))
# print(
#     'Accuracy of KNN classifier on training set: {:.2f}'.format(clf.score(data, getLabels(train_group))))
# print('Accuracy of KNN classifier on test set: {:.2f}'.format(
#     clf.score(test_group_features, test_group_labels)))

# # TODO: Linear Discriminant
# clf = LinearDiscriminantAnalysis().fit(data, getLabels(train_group))
# print(
#     'Accuracy of LDA classifier on training set: {:.2f}'.format(clf.score(data, getLabels(train_group))))
# print('Accuracy of LDA classifier on test set: {:.2f}'.format(
#     clf.score(test_group_features, test_group_labels)))
#
#
#

#
#
# names = ["3-Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]
#
# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]
#
# X_train = data
# y_train = getLabels(train_group)
# X_test = test_group_features
# y_test = test_group_labels
# # iterate over classifiers
# for name, clf in zip(names, classifiers):
#     clf.fit(X_train, y_train)
#     score_train = clf.score(X_train, y_train)
#     score_test = clf.score(X_test, y_test)
#     print('Accuracy of '+name+' on training set: '+str(score_train))
#     print('Accuracy of '+name+' on test set: '+str(score_test))
#


names = ["ID3", "3-Nearest Neighbors", "Neural Net", "AdaBoost"]

factor = {
    "ID3": 1,
    "3-Nearest Neighbors": 3,
    "Neural Net": 4,
    "AdaBoost": 3,
}

classifiers = [
    DecisionTreeClassifier(criterion="entropy"),
    KNeighborsClassifier(5),
    MLPClassifier(alpha=1),
    AdaBoostClassifier()]

X_train = data
y_train = getLabels(train_group)
X_test = test_group_features
y_test = test_group_labels
# iterate over classifiers
combine_train = {}
combine_test = {}
for i in range(len(X_test)):
    combine_train[i] = 0
    combine_test[i] = 0
for name, clf in zip(names, classifiers):
    clf_factor = factor[name]
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    for i in range(len(train_predict)):
        if train_predict[i] == True:
            combine_train[i] += 1*clf_factor
    for i in range(len(test_predict)):
        if test_predict[i] == True:
            combine_test[i] += 1*clf_factor
    # print(test_predict)
    score_train = clf.score(X_train, y_train)
    score_test = clf.score(X_test, y_test)
    print('Accuracy of ' + name + ' on training set: ' + str(score_train))
    print('Accuracy of ' + name + ' on test set: ' + str(score_test))

train_predict = []
test_predict = []
for i in range(len(combine_train)):
    if combine_train[i] >= 6:
        train_predict.append(True)
    else:
        train_predict.append(False)

    if combine_test[i] >= 6:
        test_predict.append(True)
    else:
        test_predict.append(False)

score_train = accuracy_score(train_predict, y_train)
score_test = accuracy_score(test_predict, y_test)

print('Accuracy of Combined on training set: ' + str(score_train))
print('Accuracy of Combined on test set: ' + str(score_test))

#
#
# def lookForContra(data, test):
#     hashmap={}
#     for features_list in data:
#         key = features_list[0]
#         hashmap[key] = 0
#
#     for features_list in test:
#         key = features_list[0]
#         hashmap[key] = 0
#
#     for features_list in data:
#         key = features_list[0]
#         hashmap[key] += 1
#
#     for features_list in test:
#         key = features_list[0]
#         hashmap[key] += 1
#
#     for key in hashmap:
#         if hashmap[key] > 1:
#             print(key)

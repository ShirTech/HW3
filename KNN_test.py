import csv
from classifier import knn_factory, evaluate

def KNN_test():
    results = []

    for k in [1, 3, 5, 7, 13]:
        knn_k = knn_factory(k)
        avg_accuracy, avg_error = evaluate(knn_k, 2)
        results.append([k, avg_accuracy, avg_error])


    with open("experiments6.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        for row in results:
            writer.writerow(row)

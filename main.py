import math


def distance_euclidean(list1, list2) -> float:
    sum = 0
    for x, y in zip(list1, list2):
        sum += (x-y)**2
    return math.sqrt(sum)


def main():
    # list1= [1,0,2]
    # list2=[0,2,0]
    # list1 = [1, 2, 5, 4, 4, 3, 6]
    # list2 = [3, 2, 1, 2, 1, 7, 8]
    list1=[]
    list2=[]
    list1.append(5)
    list1.append(1)
    list1.append(2)
    list2.append(1)
    list2.append(2)
    list2.append(3)
    print(list1)
    print(list2)

    list1, list2 = (list(x) for x in zip(*sorted(zip(list1, list2))))
    print(list1)
    print(list2)

main()
__author__ = 'Hugh'

def calculateSizeForAPrioriAlgorithm(M, N):
    support_threshold = 10000
    number_of_items = 1000000
    number_of_frequent_items = N
    number_of_pairs_that_appear_10000_times = 1000000
    number_of_pairs_that_appear_only_once = 2*M
    number_of_pairs_that_appear_only_once_but_conist_of_two_frequent_items = M
    number_of_bytes = 4

    first_pass = number_of_bytes* number_of_items
    second_pass = (number_of_bytes *  number_of_frequent_items) + \
                  (number_of_bytes* number_of_pairs_that_appear_only_once ** 2) - \
                  (number_of_bytes * number_of_pairs_that_appear_only_once_but_conist_of_two_frequent_items ** 2)
    return first_pass + second_pass

print("Question 1")
#N = 30,000; M = 200,000,000; S = 1,800,000,000
#N = 100,000; M = 100,000,000; S = 1,200,000,000
size1 = calculateSizeForAPrioriAlgorithm(200000000, 30000)
size2 = calculateSizeForAPrioriAlgorithm(100000000, 100000)
a = 1
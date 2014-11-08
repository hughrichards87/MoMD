__author__ = 'Hugh'
from collections import *
import numpy as np
import itertools


def lcs(a, b):
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = \
                    max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = a[x-1] + result
            x -= 1
            y -= 1
    return result

def calculateEditDistance(word1, word2):
    edit_distance = abs(len(word1) + len(word2))
    edit_distance -= 2 * len(lcs(word1, word2))

    return edit_distance


def editDistance(words):
    length = len(words)
    edit_distances = defaultdict(int)

    for i in range(0, length):
        for j in range(0, length):
            if (i != j):
                pair = [i, j]
                edit_distances["%s %s" % (words[min(pair)], words[max(pair)])] = calculateEditDistance(words[min(pair)],
                                                                                                       words[max(pair)])
    return edit_distances


def convertToRows(minhash_row, order):
    the_rows = []

    for item in minhash_row:
        the_rows.append(order[item - 1])

    return the_rows


def minhash(matrix, order):
    m = matrix.shape[1]
    minhash_row = [0] * m
    i = 1

    for r in order:
        row = matrix[r - 1]
        for c in range(0, m):
            if minhash_row[c] == 0:
                minhash_row[c] = i * row[c]
        i += 1
        if 0 not in minhash_row:
            break

    return minhash_row


def createPairs(iterable):
    # creates a combination of all the numbers in the iterable list
    return list(itertools.combinations(iterable, 2))


def getCandidatePairs(buckets):
    candidate_pairs = []

    for bucket_band in buckets:
        for key in bucket_band:
            in_the_bucket = bucket_band[key]
            if len(in_the_bucket) >= 2:
                candidate_pairs.append(createPairs(in_the_bucket))

    return candidate_pairs


def bucketHashFunction(col):
    return ', '.join(str(item) for item in col)


def localSensitiveHashing(matrix, number_of_bands):
    number_of_rows = matrix.shape[0] / number_of_bands
    number_of_cols = matrix.shape[1]
    buckets = []

    for i in range(0, number_of_bands):
        sub_matrix = matrix[i * number_of_rows:(i * number_of_rows) + number_of_rows, :]
        buckets.append(defaultdict(list))
        for x in range(0, number_of_cols):
            col = sub_matrix[:, x].tolist()
            val = bucketHashFunction(col)
            buckets[i][val].append('C%s' % (x + 1))

    return getCandidatePairs(buckets)

def calculateNumberOfDistinctShinglesInCommon(shingles_per_document):
    if len(shingles_per_document) == 0:
        return -1
    elif len(shingles_per_document) == 1:
        # only return a number
        return sum(shingles_per_document[0].values())
    elif len(shingles_per_document) == 2:
        # only return a number
        distinct_shingles_in_document1 = sorted(shingles_per_document[0])
        distinct_shingles_in_document2 = sorted(shingles_per_document[1])
        i = 0
        j = 0
        number_of_shingles_in_common = 0

        while i < len(distinct_shingles_in_document1) and j < len(distinct_shingles_in_document2):
            shingle_in_document1 = distinct_shingles_in_document1[i]
            shingle_in_document2 = distinct_shingles_in_document2[j]

            if shingle_in_document1 == shingle_in_document2:
                number_of_shingles_in_common += 1
                i += 1
                j += 1
            elif shingle_in_document1 == min(shingle_in_document1, shingle_in_document2):
                i += 1
            else:
                j += 1

        return number_of_shingles_in_common
    else:
        # return dict
        number_of_shingles_in_common = defaultdict(int)
        pairs = createPairs(range(0, len(shingles_per_document)))

        for pair in pairs:
            val = calculateNumberOfShinglesInCommon([shingles_per_document[pair[0]], shingles_per_document[pair[1]]])
            number_of_shingles_in_common[pair] = val

        return number_of_shingles_in_common

def calculateNumberOfShinglesInCommon(shingles_per_document):
    if len(shingles_per_document) == 0:
        return -1
    elif len(shingles_per_document) == 1:
        # only return a number
        return sum(shingles_per_document[0].values())
    elif len(shingles_per_document) == 2:
        # only return a number
        distinct_shingles_in_document1 = sorted(shingles_per_document[0])
        distinct_shingles_in_document2 = sorted(shingles_per_document[1])
        i = 0
        j = 0
        number_of_shingles_in_common = 0

        while i < len(distinct_shingles_in_document1) and j < len(distinct_shingles_in_document2):
            shingle_in_document1 = distinct_shingles_in_document1[i]
            shingle_in_document2 = distinct_shingles_in_document2[j]

            if shingle_in_document1 == shingle_in_document2:
                number_of_shingles_in_common += min(shingles_per_document[0][shingle_in_document1],
                                                    shingles_per_document[1][shingle_in_document2])
                i += 1
                j += 1
            elif shingle_in_document1 == min(shingle_in_document1, shingle_in_document2):
                i += 1
            else:
                j += 1

        return number_of_shingles_in_common
    else:
        # return dict
        number_of_shingles_in_common = defaultdict(int)
        pairs = createPairs(range(0, len(shingles_per_document)))

        for pair in pairs:
            val = calculateNumberOfShinglesInCommon([shingles_per_document[pair[0]], shingles_per_document[pair[1]]])
            number_of_shingles_in_common[pair] = val

        return number_of_shingles_in_common


def calculateJaccardSimilarity(shingles_per_document):
    if len(shingles_per_document) == 0:
        return -1
    elif len(shingles_per_document) == 1:
        # only return a number
        return 1
    elif len(shingles_per_document) == 2:
        # only return a number
        number_of_shingles_in_common = calculateNumberOfShinglesInCommon(shingles_per_document)
        distinct_shingles_in_document1 = sorted(shingles_per_document[0])
        distinct_shingles_in_document2 = sorted(shingles_per_document[1])
        i = 0
        j = 0
        number_of_shingles_overall = 0

        while i < len(distinct_shingles_in_document1) and j < len(distinct_shingles_in_document2):
            shingle_in_document1 = distinct_shingles_in_document1[i]
            shingle_in_document2 = distinct_shingles_in_document2[j]

            if shingle_in_document1 == shingle_in_document2:
                number_of_shingles_overall += max(shingles_per_document[0][shingle_in_document1],
                                                  shingles_per_document[1][shingle_in_document2])
                i += 1
                j += 1
            elif shingle_in_document1 == min(shingle_in_document1, shingle_in_document2):
                i += 1
            else:
                j += 1
        return number_of_shingles_in_common / number_of_shingles_overall
    else:
        # return dict
        jaccard_similarities = defaultdict(int)
        pairs = createPairs(range(0, len(shingles_per_document)))

        for pair in pairs:
            val = calculateJaccardSimilarity([shingles_per_document[pair[0]], shingles_per_document[pair[1]]])
            jaccard_similarities[pair] = val

        return jaccard_similarities

def createShingles(document, shingle_size):
    shingles = defaultdict(int)

    for i in range(0, len(document) - (shingle_size - 1)):
        shingle = document[i:i + shingle_size]
        shingles[shingle] += 1

    return shingles


def knnClustering(training_data, assignments_of_training_data, new_data, k, norm_type):
    length_of_new_data =len(new_data)
    length_of_training_data = len(training_data)
    assignments_of_new_data = [0] * length_of_new_data

    size = (length_of_training_data, length_of_new_data)
    distances = np.zeros(size)
    for i in range(0, length_of_new_data):
        new_observation = new_data[i]

        for j in range(0, length_of_training_data):
            training_observation = training_data[j]
            pair = new_observation - training_observation
            distance = np.linalg.norm(pair, norm_type)
            distances[j, i] = distance

        #time to vote
        col = distances[:, i]
        indices = np.argsort(col)
        votes = np.bincount(indices[0:k])
        if len(votes) == 1:
            assignments_of_new_data[i] = 0
        else:
            assignments_of_new_data[i] = (np.argmax(votes))

    return assignments_of_new_data

print("Question 1")
words = ['he', 'she', 'his', 'hers']
edit_distances = editDistance(words)
print(edit_distances)

print("Question 2")
matrix = np.array([[0, 1, 1, 0], \
                   [1, 0, 1, 1], \
                   [0, 1, 0, 1], \
                   [0, 0, 1, 0], \
                   [1, 0, 1, 0], \
                   [0, 1, 0, 0]])
order = [4, 6, 1, 3, 5, 2]
minhash_row = minhash(matrix, order)
rows_that_contributed = convertToRows(minhash_row, order)
print(rows_that_contributed)

print("Question 3")
matrix = np.array([[1, 2, 1, 1, 2, 5, 4], \
                   [2, 3, 4, 2, 3, 2, 2], \
                   [3, 1, 2, 3, 1, 3, 2], \
                   [4, 1, 3, 1, 2, 4, 4], \
                   [5, 2, 5, 1, 1, 5, 1], \
                   [6, 1, 6, 4, 1, 1, 4]])
number_of_bands = 3
candidate_pairs = localSensitiveHashing(matrix, number_of_bands)
print(candidate_pairs)

print("Question 4")
document1 = 'ABRACADABRA'
document2 = 'BRICABRAC'

shingles_in_document1 = createShingles(document1, 2)
number_of_shingles_in_document1 = sum(shingles_in_document1.values())
number_of_distinct_shingles_in_document1 = len(shingles_in_document1)

shingles_in_document2 = createShingles(document2, 2)
number_of_shingles_in_document2 = sum(shingles_in_document2.values())
number_of_distinct_shingles_in_document2 = len(shingles_in_document2)

number_of_distinct_shingles_in_common = calculateNumberOfDistinctShinglesInCommon([shingles_in_document1, shingles_in_document2])
jaccard_similarity = calculateJaccardSimilarity([shingles_in_document1, shingles_in_document2])
print(jaccard_similarity)

print("Question 6")
training_data = np.array([[0,0], [100, 40]])
assignments_of_training_data = [0, 1]
new_data = np.array([   [53,18], \
                        [57,5], \
                        [63,8],  \
                        [58,13], \
                    ])
k = 1
assignments_of_new_data_L1 = knnClustering(training_data, assignments_of_training_data, new_data, k, 1)
assignments_of_new_data_L2 = knnClustering(training_data, assignments_of_training_data, new_data, k, 2)
xor = np.logical_xor(np.array(assignments_of_new_data_L1) ,np.array(assignments_of_new_data_L2))
print(xor)
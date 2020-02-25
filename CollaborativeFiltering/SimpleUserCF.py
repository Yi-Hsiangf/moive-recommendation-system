# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:10:04 2018

@author: Frank
"""

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
        
testSubject = '1'
k = 3

# Load our data set and compute the user similarity matrix
ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

# Get top N similar users to our test subject
# (Alternate approach would be to select users up to some similarity threshold - try it!)
testUserInnerID = trainSet.to_inner_uid(testSubject)
similarityRow = simsMatrix[testUserInnerID]

print("train set")
print(trainSet)
print("simsMatrix")
print(simsMatrix)
print(simsMatrix.shape)
print("testUserInnerID")
print(testUserInnerID)
print("similarityRow")
print(similarityRow)
similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )

kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])
print("kNeighbors : " + str(kNeighbors))
# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    print("similarUser: " + str(similarUser))
    innerID = similarUser[0]
    print("innerID: " + str(innerID))
    userSimilarityScore = similarUser[1]
    print("userSimilarityScore: " + str(userSimilarityScore))
    theirRatings = trainSet.ur[innerID]
    print("theirRatings: " + str(theirRatings))
    for rating in theirRatings:
        print("rating[0]: " + str(rating[0]))
        print("rating[1]: " + str(rating[1]))
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
    
print(candidates)
# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1
    
# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(ml.getMovieName(int(movieID)), ratingSum)
        pos += 1
        if (pos > 10):
            break




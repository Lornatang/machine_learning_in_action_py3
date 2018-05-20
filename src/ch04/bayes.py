"""
Create by 2018-05-20

@author: Shiyipaisizuo
"""

from numpy import *


def load_data_set():
    postinglist = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                    'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 1 is abusive, 0 not
    classvec = [0, 1, 0, 1, 0, 1]
    return postinglist, classvec


def create_vocab_list(dataset):
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set(document)

    return list(vocabset)


def set_of_word_vec(vocablist, inputset):
    returnvec = [0]*len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1
        else:
            print("The word:{} is not in my vocalulary!").format(word)

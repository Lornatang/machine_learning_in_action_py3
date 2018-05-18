"""
crete by 2018-05-18

@author: Shiyipaisizuo
"""

from numpy import *
print(random.rand(4, 4))
randMat = mat(random.rand(4, 4))
print(randMat.I)
invRandMat = randMat.I
myEye = randMat * invRandMat
print(myEye - eye(4))

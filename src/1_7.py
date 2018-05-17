from numpy import *
random.rand(4, 4)
randMat = mat(random.rand(4, 4))
randMat.I
invRandMat = randMat.I
myEye = randMat * invRandMat
myEye - eye(4)

import numpy as np
import random

def softmax(x):
    assert len(x.shape) <= 2
    #axis=0是列，axis=1是行
    #axis=0是指第一个维度，axis=1是指第二个维度 所谓的第几个维度是x.shape()的结果
    #softmax满足所有元素相加=1
    y = np.exp(x - np.max(x, axis=len(x.shape) - 1, keepdims=True))
    #print(np.max(x, axis=0, keepdims=True))
    #print(y)
    normalization = np.sum(y, axis=len(x.shape) - 1, keepdims=True)
    #print(normalization)
    return np.divide(y, normalization)

def test_softmax_basic():
    print("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print(test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print("You should verify these results!\n")

def test_softmax():
    print("Running your tests...")

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()

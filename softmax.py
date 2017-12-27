import numpy as np
import random

def softmax(x):
    assert len(x.shape) <= 2
    #axis=0是列，axis=1是行
    #axis=0是指第一个维度，axis=1是指第二个维度 所谓的第几个维度是x.shape()的结果
    #softmax满足所有元素相加=1
    #使用性质，softmax不会随着输入向量偏移而偏移
    y = np.exp(x - np.max(x, axis=len(x.shape) - 1, keepdims=True))
    #print(np.max(x, axis=0, keepdims=True))
    #print(y)
    normalization = np.sum(y, axis=len(x.shape) - 1, keepdims=True)
    #print(normalization)
    return np.divide(y, normalization)

def softmax__(x):
    assert len(x.shape)>=2
    x-=np.max(x,axis=1,keepdims=True)
    print(x)
    x=np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
    return x

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
    test=softmax__(np.array([[1001,1002,1003],[18,5,100]]))
    print(test)

if __name__ == "__main__":
    #test_softmax_basic()
    test_softmax()

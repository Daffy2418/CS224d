import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))#10*5
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))#1*5
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))#5*10
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy)) #1*10

    ### forward propagation
    h_per_item = sigmoid(np.dot(data, W1) + b1)
    yhat_per_item = softmax(np.dot(h_per_item, W2) + b2)
    ### cost function loss
    cost = -np.sum(labels * np.log(yhat_per_item))

    ### backward propagation
    grad_softmax_per_item = yhat_per_item - labels
    grad_b2 = np.sum(grad_softmax_per_item, axis=0, keepdims=True)
    grad_W2 = np.dot(h_per_item.T, grad_softmax_per_item)
    grad_sigmoid_per_item = sigmoid_grad(h_per_item)
    grad_b1_per_item = np.dot(grad_softmax_per_item, W2.T) * grad_sigmoid_per_item
    grad_b1 = np.sum(grad_b1_per_item, axis=0, keepdims=True)
    grad_W1 = np.dot(data.T, grad_b1_per_item)

    assert grad_b2.shape == b2.shape
    assert grad_W2.shape == W2.shape
    assert grad_b1.shape == b1.shape
    assert grad_W1.shape == W1.shape
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((grad_W1.flatten(), grad_b1.flatten(), 
        grad_W2.flatten(), grad_b2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # 20*10
    labels = np.zeros((N, dimensions[2])) #20*10
    for i in xrange(N): # each row has 1 positive,9 neagtive
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )#115 total parameter in neual network

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

if __name__ == "__main__":
    sanity_check()

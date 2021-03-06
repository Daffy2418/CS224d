import numpy as np
from scipy.special import expit

def sigmoid(x):
    #logistic function
    #expit(x)=1/(1+exp(-x))
    return expit(x)

def sigmoid_grad(f):
    #f'(x)=f(x)(1-f(x))
    return f - f * f

def test_sigmoid_basic():
    print("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print(f)
    assert np.amax(f - np.array([[0.73105858, 0.88079708], 
        [0.26894142, 0.11920292]])) <= 1e-6
    print(g)
    assert np.amax(g - np.array([[0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])) <= 1e-6
    print("You should verify these results!\n")

if __name__ == "__main__":
    test_sigmoid_basic();

import numpy as np
import random

from sigmoid import sigmoid, sigmoid_grad

def gradcheck_naive(f_and_grad, x):
    """ 
    Gradient check for a function f
    - f_and_grad should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 
    #internal state, make the random value same
    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f_and_grad(x) # fx=sum(x**2)  grad=2x

    y = np.copy(x)
    # Iterate over all indexes in x
    # return [row,col]
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        
        h=1e-6
        
        x[ix]+=h
        random.setstate(rndstate)
        plus_h_fx,plus_h_grad=f_and_grad(x)
        random.setstate(rndstate)
        x[ix]-=2.*h
        minus_h_fx,minus_h_grad=f_and_grad(x)
        numgrad=(plus_h_fx-minus_h_fx)/(2*h)
        
        reldiff=abs(numgrad-grad[ix])#/max(1, abs(numgrad), abs(grad[ix]))
        
        # Compare gradients
        print 'reldiff', reldiff
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
    
        it.iternext() # Step to next dimension

    print "Gradient check passed!"

def sanity_check():
    #function
    quad_and_grad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad_and_grad, np.array(123.456))      # scalar test
    gradcheck_naive(quad_and_grad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad_and_grad, np.random.randn(4,5))   # 2-D test

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    sigmoid_and_grad = lambda x: (np.sum(sigmoid(x)), sigmoid_grad(sigmoid(x)))
    gradcheck_naive(sigmoid_and_grad, np.array(1.23456))      # scalar test
    gradcheck_naive(sigmoid_and_grad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(sigmoid_and_grad, np.random.randn(4,5))   # 2-D test
    gradcheck_naive(sigmoid_and_grad, np.arange(-5.0, 5.0, 0.1))   # range test
    sincos_and_grad = lambda x: (np.sin(x) + np.cos(x), np.cos(x) - np.sin(x))

    gradcheck_naive(sincos_and_grad, np.array(1.0))

    print

if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()

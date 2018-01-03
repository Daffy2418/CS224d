import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

def generate_dataset():
    #y=2x+e  e is sampled from a normal distribution
    x_batch=np.linspace(-1,1,101)
    #np.random.randn(*x_batch.shape)
    y_batch=2*x_batch+np.random.randn(*x_batch.shape)*0.3
    return x_batch,y_batch
    
def linear_regression():
    #placeholder
    # Note: The second argument shape=(None,) indicates that 
    # these variables take on a 1-dimensional value of a dynamic 
    # size. We can use the None value in this case to allow for 
    # arbitrary batch sizes.
    x=tf.placeholder(tf.float32,shape=(None,),name='x')
    y=tf.placeholder(tf.float32,shape=(None,),name='y')
    #variable sharing
    with tf.varibale_scope('lreg') as scope:
        w=tf.Variable(np.random.normal(),name='W')
        y_pred=tf.multiply(w,x)
        
        loss=tf.reduce_mean(tf.square(y_pred-y))
    return x,y,y_pred,loss
    
def run():
    x_batch,y_batch=generate_dataset()
    x,y,y_pred,loss=linear_regression()
    optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    ##must initialize the variable
    init=tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        
        feed_dict={x:x_batch,y:y_batch}
        for _ in range(30):
            loss_val,_=session.run([loss,optimizer],feed_dict)
            print("loss:",loss_val)
        
        y_pred_batch=session.run(y_pred,{x:x_batch})
    
    plt.figure(1)
    plt.scatter(x_batch,y_batch)
    plt.plot(x_batch,y_pred_batch)
    
if __name__=='__main__':
    run()

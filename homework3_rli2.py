#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Python version 3.7.0
Author: Ruojun
'''
import numpy as np


# In[2]:


'''
Loading The Train,Test,Validation from local directory
'''
trainX = np.load("mnist_train_images.npy")#[:55000,:784]
trainY = np.argmax(np.load("mnist_train_labels.npy"),axis=1)#[:55000,:]
testX = np.load("mnist_test_images.npy")#[:5000,:784]
testY = np.argmax(np.load("mnist_test_labels.npy"),axis=1)#[:5000,:]
validationX = np.load("mnist_validation_images.npy")#[:10000,:784]
validationY = np.argmax(np.load("mnist_validation_labels.npy"),axis=1)#[:10000,:]
'''
Printing the dataset shape
'''
print("The shape of trainX :", trainX.shape)
print("The shape of trainY :", trainY.shape)
print("The shape of validationX :", validationX.shape)
print("The shape of ValidationY :", validationY.shape)
print("The shape of testX :", testX.shape)
print("The shape of testY :", testY.shape)


# In[3]:


def compute_z(x,W):
    '''
        Compute the linear logit values of a data instance. z =  W x
        Input:
            x: the feature vector of a data instance, a float numpy matrix of shape p by 1. Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
        Output:
            z: the linear logits, a float numpy vector of shape c by 1. 
    '''
    #print(W.shape,x.shape)
    z = W*x
    return z 


# In[4]:


def compute_a(z):
    '''
        Compute the softmax activations.
        Input:
            z: the logit values of softmax regression, a float numpy vector of shape c by 1. Here c is the number of classes
        Output:
            a: the softmax activations, a float numpy vector of shape c by 1. 
    '''
    a = np.exp(z) / sum(np.exp(z))
    return a


# In[5]:


def compute_L(a,y,alpha=0.1):
    '''
        Compute multi-class cross entropy, which is the loss function of softmax regression. 
        Input:
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes. 
            y: the label of a training instance, integer value, 0-c-1.
        Output:
            L: the loss value of softmax regression, a float scalar.
    '''
    a_max = a[y]
    L = np.float(-y*np.log(a_max) + alpha/2*sum(a.T*a))
    return L 


# In[6]:


def compute_dz_dW(x,c):
    '''
        Compute local gradient of the logits function z w.r.t. the weights W.
        Input:
            x: the feature vector of a data instance, a float numpy vector of shape p by 1. Here p is the number of features/dimensions.
            c: the number of classes, an integer. 
        Output:
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix, a numpy float matrix of shape (c by p). 
                   The (i,j)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
    '''
    dz_dW = np.repeat(x,c,axis=1).transpose()
    return dz_dW
def compute_da_dz(a):
    '''
        Compute local gradient of the softmax activations a w.r.t. the logits z.
        Input:
            a: the activation values of softmax function, a numpy float vector of shape c by 1. Here c is the number of classes.
        Output:
            da_dz: the local gradient of the activations a w.r.t. the logits z, a float numpy matrix of shape (c by c). 
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
    '''
    classes = a.shape[0]
    da_dz = np.asmatrix(np.zeros((classes,classes)))
    for i in range(classes):
        for j in range(classes):
            if i==j:
                da_dz[i,j] = a[i]*(1 - a[j])
            else:
                da_dz[i,j] = -a[i]*a[j]
    return da_dz 
def compute_dL_da(a, y,alpha):
    '''
        Compute local gradient of the multi-class cross-entropy loss function w.r.t. the activations.
        Input:
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes. 
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        Output:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape c by 1. 
                   The i-th element dL_da[i] represents the partial gradient of the loss function w.r.t. the i-th activation a[i]:  d_L / d_a[i].
    '''
    dL_da = np.asmatrix(np.zeros(a.shape))
    if a[y]==0:
        dL_da[y,0] = -1e6
    else:
        dL_da[y,0] = -1/a[y] + alpha*sum(a.T*a)
    return dL_da

def compute_dL_dz(dL_da,da_dz):
    '''
       Given the local gradients, compute the gradient of the loss function L w.r.t. the logits z using chain rule.
        Input:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape c by 1. 
                   The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i]:  d_L / d_a[i].
            da_dz: the local gradient of the activation w.r.t. the logits z, a float numpy matrix of shape (c by c). 
                   The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] )
        Output:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape c by 1. 
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
    '''
    dL_dz = np.dot(da_dz,dL_da)
    return dL_dz
def compute_dL_dW(dL_dz,dz_dW):
    '''
       Given the local gradients, compute the gradient of the loss function L w.r.t. the weights W using chain rule. 
        Input:
            dL_dz: the gradient of the loss function L w.r.t. the logits z, a numpy float vector of shape c by 1. 
                   The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th logit z[i]:  d_L / d_z[i].
            dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a numpy float matrix of shape (c by p). 
                   The i,j -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[i,j]:   d_z[i] / d_W[i,j]
        Output:
            dL_dW: the global gradient of the loss function w.r.t. the weight matrix, a numpy float matrix of shape (c by p). 
                   Here c is the number of classes.
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
        Hint: you could solve this problem using 2 lines of code
    '''
    dL_dW = np.matrix(np.asarray(dL_dz)*np.asarray(dz_dW))
    return dL_dW


# In[7]:


def update_W(W, dL_dW, lr=0.001):
    '''
       Update the weights W using gradient descent.
        Input:
            W: the current weight matrix, a float numpy matrix of shape (c by p). Here c is the number of classes.
            alpha: the step-size parameter of gradient descent, a float scalar.
            dL_dW: the global gradient of the loss function w.r.t. the weight matrix, a numpy float matrix of shape (c by p). 
                   The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j]:  d_L / d_W[i,j]
        Output:
            W: the updated weight matrix, a numpy float matrix of shape (c by p).
        Hint: you could solve this problem using 1 line of code 
    '''
    W -= dL_dW*lr
    return W


# In[33]:


#--------------------------
# train
def train(X, Y,lr=0.01, n_epoch=10,alpha=0.1,mini_batch_size=30):
    '''
       Given a training dataset, train the softmax regression model by iteratively updating the weights W and biases b using the gradients computed over each data instance. 
        Input:
            X: the feature matrix of training instances, a float numpy matrix of shape (n by p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
            Y: the labels of training instance, a numpy integer numpy array of length n. The values can be 0 or 1.
            lr: the step-size parameter of gradient ascent, a float scalar.
            n_epoch: the number of passes to go through the training set, an integer scalar.
            aplha: the regularization strength
            mini_size: mini-batch size, when mini_size = 1, it's SGD.
        Output:
            W: the weight matrix trained on the training set, a numpy float matrix of shape (c by p).
    '''
    #number of samples:
    n = X.shape[0]
    # number of features
    p = X.shape[1]
    # number of classes 
    c = 10
    # randomly initialize W and b
    W = np.asmatrix(np.random.rand(c,p))
    dL_dW = []
    training_data = list(zip(X, Y))
    for _ in range(n_epoch):
        # go through each training instance
        mini_batches = [ training_data[k:k + mini_batch_size]
                               for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            for x,y in mini_batch:
                x = np.reshape(x,(p,1))
                # Forward pass: compute the logits, softmax and cross_entropy 
                z = compute_z(x,W)
                a = compute_a(z)
                L = compute_L(a,y,alpha)
                # Back Propagation: compute local gradients of cross_entropy, softmax
                dz_dW = compute_dz_dW(x, c)
                da_dz = compute_da_dz(a)
                dL_da = compute_dL_da(a, y,alpha)
                dL_dz = compute_dL_dz(dL_da,da_dz)
                dL_dW.append(compute_dL_dW(dL_dz, dz_dW))
            #Update of mini-batch
            W = update_W(W,np.mean(dL_dW),lr)
            dL_dW = []
        print('Epoch'+str(_),'Training Loss:',L)
    return W


# In[34]:


def predict(Xtest,Ytest,W,alpha=0.1):
    '''
       Predict the labels of the instances in a test dataset using softmax regression.
        Input:
            Xtest: the feature matrix of testing instances, a float numpy matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            W: the weight vector of the logistic model, a float numpy matrix of shape (c by p).
            b: the bias values of the softmax regression model, a float vector of shape c by 1.
        Output:
            Y: the predicted labels of test data, an integer numpy array of length ntest Each element can be 0, 1, ..., or (c-1) 
            P: the predicted probabilities of test data to be in different classes, a float numpy matrix of shape (ntest,c). Each (i,j) element is between 0 and 1, indicating the probability of the i-th instance having the j-th class label. 
    '''
    n = Xtest.shape[0]
    p = Xtest.shape[1]
    c = W.shape[0]
    Y = np.zeros(n) # initialize as all zeros
    P = np.asmatrix(np.zeros((n,c)))  
    L = []
    acc=0
    for i, (x,y) in enumerate(zip(Xtest,Ytest)):
        x = np.reshape(x,(p,1))
        P = compute_a(compute_z(x,W))
        y_hat = np.argmax(P)
        if y==y_hat:acc=acc+1
        L.append(compute_L(P,y,alpha))
    L = np.mean(L)
    acc =acc/n*100
    print('test_entropy_loss:', L)
    print('test_class_loss:',acc,'%')
    
    return Y,P,L


# In[35]:


W= train(trainX,trainY,mini_batch_size=1)


# In[36]:


predict(testX,testY,W)


# In[ ]:


W= train(trainX,trainY,mini_size=1)


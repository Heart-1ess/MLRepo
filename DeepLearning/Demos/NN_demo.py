import numpy as np

from Sigmoid_demo import sigmoid

def initialize_with_zeros(shape):
    '''
    Create (shape, 1) like W and b = 0
    
    input: 
        shape: Shape of the vectors.
    
    output:
        w: Parameter matrix w.
        b: Offset b.
    '''

    w = np.zeros((shape, 1))
    b = 0
    
    assert(w.shape == (shape, 1))
    assert(isinstance(b ,float) or isinstance(b ,int))
    
    return w, b

def propagate(w, b, X, Y):
    '''
    Forward and backward propagation.
    
    input:
        w: Parameter matrix w.
        b: Offset b.
        X: Raw data.
        Y: Label.
        
    output:
        grads: Gradients of the current epoch.
        loss_avg: Average loss calculated by loss function.
    '''
    
    # Forward
    # w(n, 1), X(n, m), b(1, 1) -> A(1, m)
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    
    # Loss function
    # loss: L = 1/2 * (yhat - y)^2
    # loss for sigmoid: L = -(ylog(yhat)) - (1 - y)log(1 - yhat)
    # loss avg = 1/m * sum(L)
    m = X.shape[1]
    L = -(Y * np.log(A)) - (1 - Y) * np.log(1 - A)
    loss_avg = 1 / m * L
    
    # Gradient calculate
    dz = A - Y
    dw = 1 / m * np.dot(X, dz.T)
    db = 1 / m * np.sum(dz)
    
    # Backward
    
    # output
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    loss_avg = np.squeeze(loss_avg)
    assert(loss_avg.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, loss_avg

def optim(w, b, X, Y, num_iterations, learning_rate):
    '''
    Optimize the model.
    
    input:
        w: Parameter matrix w.
        b: Offset b.
        X: Raw data.
        Y: Label.
        num_iterations: Numbers of epochs to iterate.
        learning_rate: Weight of learning, decides how fast the model iterates.
        
    output:
        params: Final parameters after iteration.
        grads: Gradient of the last epoch.
        loss_avgs: Average loss calculated by loss function.
    '''
    
    loss_avgs = []
    
    for i in range(0, num_iterations):
        
        # Calculate grads in propagation.
        grads, loss_avg = propagate(w, b, X, Y)
        
        dw = grads['dw']
        db = grads['db']
        
        # Update the params.
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Log the loss every 100 steps.
        if i % 100 == 0:
            loss_avgs.append(loss_avg)
        if i % 100 == 0:
            print("Current epochs: {}\nCurrent loss: {}".format(i, loss_avg))
        
    params = {
        "w": w,
        "b": b
    }
    
    grads = {
        "dw": dw,
        "db": db
    }
    
    return params, grads, loss_avgs
    
def predict(w, b, X):
    '''
    Do predict, 1 or 0 task.
    
    input:
        w: Parameter matrix w.
        b: Offset b.
        X: Data need to predict. 
        
    output:
        Y_pred: Predictions from X.
    '''
    
    # Confirm the shapes meet the need.
    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    w.reshape(X.shape[0], 1)
    
    # Go through the network.
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    
    for i in range(A.shape[1]):
        # 1 or 0
        if A[0, i] <= 0.5:
            Y_pred[0, i] = 0
        else:
            Y_pred[0, i] = 1
            
    assert(Y_pred.shape == (1, m))
    
    return Y_pred

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):
    '''
    Simulate how the model works.
    
    input:
        X_train: Raw data for training.
        Y_train: Labels for training.
        X_test: Raw data for testing.
        Y_test: Labels for testing.
        num_iterations: Numbers of epochs to iterate.
        learning_rate: Weight of learning, decides how fast the model iterates.
    
    output:

    '''
    # Confirm the shapes equal.
    assert(X_train.shape == X_test.shape)
    assert(Y_train.shape == Y_test.shape)
    
    # Get shape of X.
    n = X_train.shape[0]
    
    # Initialize.
    w, b = initialize_with_zeros(n)
    
    # Train.
    params, grads, loss_avgs = optim(w, b, X_train, Y_train, num_iterations=num_iterations, learning_rate=learning_rate)
    
    # Get params.
    w = params['w']
    b = params['b']
    
    # Predict.
    y_pred_train = predict(w, b, X_train)
    y_pred_test = predict(w, b, X_test)
    
    # Print accuracy, 
    print("Accuracy on training set: {}".format(100 - np.mean(np.abs(y_pred_train - Y_train)) * 100))
    print("Accuracy on testing set: {}".format(100 - np.mean(np.abs(y_pred_test - Y_test)) * 100))
    
    d = {
        "Loss_avg": loss_avgs,
        "Y_prediction_test": y_pred_test,
        "Y_prediction_train": y_pred_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    
    return d
    
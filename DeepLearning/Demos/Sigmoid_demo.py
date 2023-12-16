import numpy as np

def sigmoid(z):
    '''
    Sigmoid.
    '''
    res = 1 / (1 + np.exp(-z))
    return res

def sigmoid_derivation(z):
    '''
    Derivation of sigmoid.
    
    (sigmoid(x))' = sigmoid(x)(1 - sigmoid(x))
    '''
    ds = sigmoid(z) * (1 - sigmoid(z))
    return ds
    
if __name__ == "__main__":
    z = np.array([1, 2, 3])
    print(sigmoid(z))
    print(sigmoid_derivation(z))
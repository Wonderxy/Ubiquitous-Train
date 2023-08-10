import numpy as np
import tensorly as tl

def padding_tensor(rank,ttr):
    eM = np.eye(ttr)
    eT = tl.tensor(np.zeros((ttr,rank,ttr)))
    for i in range(rank):
        eT[:,i,:] = eM 
    return eT

if __name__ == '__main__':
    t = padding_tensor(5,3)
    print(t.shape)
    for i in range(5):
        print(t[:,i,:])
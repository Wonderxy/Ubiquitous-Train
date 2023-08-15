import pickle
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
from tt.tt_tensor import TTTensor
import tensorly as tl
import decomposition.ttd as ttd 
import numpy as np

path = 'D:/Files/VisualStudioCode/TT2.0/Ubiquitous-Train/dataset'

def stroe_tensor(tensorList,tensorNameList,path):
    """Store in tensor form

    Parameters
    ----------
    tensorList:List of tensors that need to be stored
    tensorNameList:Corresponding name

    Returns
    -------
    None
    """
    tNum = len(tensorList)
    tNameNum = len(tensorNameList)
    if tNum != tNameNum:
        raise ValueError("The elements in tensorList and tensorNameList should correspond one by one")
    for i in range(tNum):
        with open(path+tensorNameList[i]+".bin", 'wb+') as fp: 
            pickle.dump(tensorList[i], fp)


def store_tt(ttList,ttNameList,path):
    """Store in tt form

    Parameters
    ----------
    ttList:List of tts that need to be stored
    ttNameList:Corresponding name

    Returns
    -------
    None
    """
    ttNum = len(ttList)
    ttNameNum = len(ttNameList)
    if ttNum != ttNameNum:
        raise ValueError("The elements in tensorList and tensorNameList should correspond one by one")
    for i in range(ttNum):
        if isinstance(ttList[i],TTTensor):
            with open(path+ttNameList[i]+".bin", 'wb+') as fp: 
                pickle.dump(ttList[i], fp)
        else:
            raise ValueError("Please enter data of type TTTensor")


def load_tensor(tensorNameList,path):
    """Load tensor
    Parameters
    ----------
    tensorNameList:List of name of stored tensors

    Returns
    -------
    List of corresponding tensors
    """
    tNameNum = len(tensorNameList)
    tensorList = []
    for i in range(tNameNum):
        with open(path+tensorNameList[i]+".bin", 'rb') as fp:
            t = pickle.load(fp)
            tensorList.append(t)
    return tensorList


def load_tt(ttNameList,path):
    """Load tt
    Parameters
    ----------
    ttNameList:List of name of stored tensors

    Returns
    -------
    List of corresponding tts
    """
    tNameNum = len(ttNameList)
    ttList = []
    for i in range(tNameNum):
        with open(path+ttNameList[i]+".bin", 'rb') as fp:
            t = pickle.load(fp)
            ttList.append(t)
    return ttList


if __name__ == "__main__":
    t1 = tl.tensor(np.random.randint(0,2,(2,2,3,4,1000,6),dtype="int"))
    t2 = tl.tensor(np.random.randint(0,2,(2,3,1000,6,7),dtype="int"))
    t3 = tl.tensor(np.random.randint(0,2,(2,3,6,5),dtype="int")) 
    t4 = tl.tensor(np.random.randint(0,2,(2,2,1000,2),dtype="int"))
    tList = [t1,t2,t3,t4]
    ttList = []
    for t in tList:
        tt = ttd.TensorTrain(rank=0,method="tt_svd").fit_transform(t) 
        ttList.append(tt)
    store_tt(ttList,["t1","t2","t3","t4"],path+"/tt_storage/")
    
    newttList = load_tt(["t1","t2","t3","t4"],path+"/tt_storage/")
    for i in range(len(newttList)):
        print(newttList[i].shape)
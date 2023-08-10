import tensorly as  tl
import numpy as np
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
from utils.forList import factorial_list
import tensor.contraction as tc
import copy
import time
from multiprocessing import Pool
import decomposition.ttd as ttd 
import tensor.mathematical as tm
from tt.tt_tensor import TTTensor

def validate_join_tt(ttList,toList,corList):
    """Verify before operation, including the quantity of tts and
        the problem of whether the corresponding dimensions between corresponding tts match

    Parameters
    ----------
    ttList : list[[[ttcore1],[ttcore2],...],[],...], tt List
    toList : list[], tensor corresponding to join together
    corList : list[[],[[join_orders],[join_orders_main]],[[],[]],...],The relationship between corresponding orders

    Returns
    -------
    shpList : list[(shape)], shape List
    lenList : list[], length List
    """
    #validate_tensor_num
    tNum = len(ttList)
    if len(toList) != tNum or len(corList) != tNum:
        raise ValueError("The number of parameters for tList,toList,corList needs to be equal")
    #validate_join_order
    shpList = []
    rankList = []
    for t in ttList:
        shpList.append(t.shape)
        rankList.append(t.rank)
    
    for i in range(1,tNum):
        toTensor = toList[i]
        for j in range(len(corList[i][0])):
            if shpList[i][corList[i][0][j]] != shpList[toTensor][corList[i][1][j]]:
                raise ValueError("Mismatch in the number of ordered dimensions between corresponding tensors")
    
    lenList = []
    for shp in shpList:
        lenList.append(len(shp))
    
    return shpList,rankList,lenList

def padding_tensor(order,ttr):
    """Generate filled matrices corresponding to different ttranks

    Parameters
    ----------
    order : int, Corresponding to the order in the original tensor
    ttr : int, The rank of core tensors tt with adjacency

    Returns
    -------
    eT : ndarray/tl.tensor
        Padding tensor composed of identity matrix
    """
    eM = np.eye(ttr)
    eT = tl.tensor(np.zeros((ttr,order,ttr)))
    for i in range(order):
        eT[:,i,:] = eM 
    return eT

def padding(ttList,FinalOrderList,toList,corList,shpList):
    FinalPaddingList = []
    for i in range(len(ttList)):
        paddingList = []
        start = 0
        for j in range(len(FinalOrderList)):
            flag = True
            joinOrderList = tc.join_order(FinalOrderList[j][0],FinalOrderList[j][1],toList,corList)
            for k in range(start,len(ttList[i].factors)):
                if ([i,k] in joinOrderList) or ([i,k] == FinalOrderList[j]):
                    paddingList.append(ttList[i][k])
                    start += 1
                    flag = False
                    break
            if flag:
                order = shpList[FinalOrderList[j][0]][FinalOrderList[j][1]]
                ttr = (ttList[i].rank)[start]
                paddingList.append(padding_tensor(order,ttr))
            
        FinalPaddingList.append(paddingList)
    return FinalPaddingList


def tt_join(ttList,toList,corList):
    """Tensor based multi tensor join operation

    Parameters
    ----------
    tList : ndarray/tl.tensor[ndarray/tl.tensor], tensorList
    toList : list[list], tensor corresponding to join together
    corList : list, Tensor corresponding to join

    Returns
    -------
    joinTensor : ndarray/tl.tensor
        The Result of Joining Multiple Tensors
    """
    shpList,rankList,lenList = validate_join_tt(ttList,toList,corList)
    join_tree = tc.create_tree(toList,lenList,corList)
    join_tree.show()
    FinalOrderList = tc.tree_join(join_tree.get_node(join_tree.root),join_tree)
    print("FinalOrderList:",FinalOrderList)

    joinTensorShape = []
    for site in FinalOrderList:
        joinTensorShape.append(shpList[site[0]][site[1]])
    print("joinTensorShape:",joinTensorShape)

    FinalPaddingList = padding(ttList,FinalOrderList,toList,corList,shpList)
    factorList = []
    for i in range(len(FinalPaddingList[0])):
        factor = FinalPaddingList[0][i]
        for j in range(1,len(FinalPaddingList)):
            factor = tm.factors_kron(FinalPaddingList[j][i],factor)
        factorList.append(factor)

    return TTTensor(factorList)


    

if __name__ == "__main__":
    t1 = tl.tensor(np.random.randint(0,2,(2,2,3,4,5,6)))
    t2 = tl.tensor(np.random.randint(0,2,(2,3,5,6,7)))
    t3 = tl.tensor(np.random.randint(0,2,(2,3,6,5))) 
    t4 = tl.tensor(np.random.randint(0,2,(2,2,5,2)))
    # t5 = tl.tensor(np.random.randint(0,2,(2,3,6,2)))
    # tList = [t1,t2,t3,t4,t5]
    # toList = [0,0,1,0,1]
    # corList = [[],[[1,2],[2,4]],[[1,2],[1,3]],[[1,2],[1,4]],[[1,2],[1,3]]]
    tList = [t1,t2,t3,t4]
    toList = [0,0,1,0]
    corList = [[],[[1,2],[2,4]],[[1,2],[1,3]],[[1,2],[1,4]]]
    ttList = []
    for t in tList:
        tt = ttd.TensorTrain(rank=0,method="tt_svd").fit_transform(t) 
        ttList.append(tt)
    time1 = time.time()
    t6 = tt_join(ttList,toList,corList)
    time2 = time.time()
    print("Time:",time2-time1)
    print(t6.to_tensor().shape)

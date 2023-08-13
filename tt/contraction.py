import tensorly as  tl
import numpy as np
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
from utils.forList import factorial_list
import tensor.contraction as tc
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
    """Fill the original tensor train with padding tensors to ensure that 
        it can perform Kron multiplication of the corresponding order

    Parameters
    ----------
    ttList : list[TTTensor], ttlist
    FinalOrderList : list[[tensor,order]]
    toList : list[], tensor corresponding to join together
    corList : list[[order,order],[order,order]],...], Tensor corresponding to join
    shpList : list, shape list
    
    Returns
    -------
    FinalPaddingList : list[tl.tensor,tl.tensor,...]
        The tensor train of each new tensor after filling in the unit padding tensor
    """
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
    """TT based multi tensor join operation

    Parameters
    ----------
    ttList : list[TTTensor], ttlist
    toList : list[list], tensor corresponding to join together
    corList : list, Tensor corresponding to join

    Returns
    -------
    factorList : list[TTTensor]
        The Result of Joining Multiple Tensors baseed on TT
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
            factor = tm.factors_kron(factor,FinalPaddingList[j][i])#
        factorList.append(factor)

    return TTTensor(factorList)

"""
    Optimize for excessive dimensionality caused by excessive padding tensors in tt-join,
    otherwise it will lead to an increase in spatial/temporal complexity
"""

def padding_opt(ttList,FinalOrderList,toList,corList,shpList):
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
                paddingList.append(ttr)#padding_tensor(order,ttr)-->ttr
        FinalPaddingList.append(paddingList)
    return FinalPaddingList


def kron_block(matrix,blocknum,ttr):
    row_b = blocknum[0]
    col_b = blocknum[1]
    col_M = np.split(matrix,col_b,axis=1)
    splited_list = []
    for m in col_M:
        row_M = np.split(m,row_b,axis=0)
        splited_list.append(row_M)
    for i in range(col_b):
        for j in range(row_b):
            splited_list[i][j] = np.kron(np.eye(ttr),splited_list[i][j])
    colList = []
    for i in range(col_b):
        rowList = []
        for j in range(row_b):
            rowList.append([splited_list[i][j]])
        colList.append(np.block(rowList))
    finalM = np.block(colList)
    print("finalM:",finalM.shape)
    return finalM

def tt_join_opt(ttList,toList,corList):
    shpList,rankList,lenList = validate_join_tt(ttList,toList,corList)
    join_tree = tc.create_tree(toList,lenList,corList)
    join_tree.show()
    FinalOrderList = tc.tree_join(join_tree.get_node(join_tree.root),join_tree)
    print("FinalOrderList:",FinalOrderList)

    joinTensorShape = []
    for site in FinalOrderList:
        joinTensorShape.append(shpList[site[0]][site[1]])
    print("joinTensorShape:",joinTensorShape)

    FinalPaddingList = padding_opt(ttList,FinalOrderList,toList,corList,shpList)#padding-->padding_opt
    
        
    factorList = []
    for i in range(len(FinalPaddingList[0])):
        tNum = 0;ttr = 1
        rowNum = 1; colNum = 1
        factor1List = []
        for j in range(0,len(FinalPaddingList)):
            if isinstance(FinalPaddingList[j][i],int):
                ttr *= FinalPaddingList[j][i]
            elif tNum == 0:
                factor = FinalPaddingList[j][i]
                factor1List.append([ttr,rowNum,colNum])
                tNum += 1; ttr = 1
                rowNum = factor.shape[0];colNum = factor.shape[2]
            else:
                factor = tm.factors_kron(factor,FinalPaddingList[j][i])
                factor1List.append([ttr,rowNum,colNum])
                ttr = 1
                rowNum = factor.shape[0];colNum = factor.shape[2]#ERROR
        factor1List.append([ttr])
        factor1List.insert(0,factor)    
        factorList.append(factor1List)
        print("[factor,ttr,blockNum]",factor1List[0].shape,factor1List[1:])
    
    FinanFactorList = []
    for factorttr in factorList:
        factor = factorttr[0]
        for i in range(1,len(factorttr)-1):
            print("i",i)
            newfactor = np.zeros((factor.shape[0]*factorttr[i][0],factor.shape[1],factor.shape[2]*factorttr[i][0]))
            print("newfactor:",newfactor.shape)
            for j in range(factor.shape[1]):
                #Block the matrix
                print("row col",(factorttr[i][1],factorttr[i][2]))
                newfactor[:,j,:] = kron_block(factor[:,j,:],(factorttr[i][1],factorttr[i][2]),factorttr[i][0])
            factor = newfactor
            print("factor:",factor.shape)

        newfactor = np.zeros((factor.shape[0]*factorttr[-1][0],factor.shape[1],factor.shape[2]*factorttr[-1][0]))
        
        for j in range(factor.shape[1]):
            #Block the matrix
            newfactor[:,j,:] = np.kron(factor[:,j,:],np.eye(factorttr[-1][0]))
        FinanFactorList.append(newfactor)

    return TTTensor(FinanFactorList)

"""def tt_contract(factorttrA,factorttrB):
    factorA = factorttrA[0]
    ttrA = factorttrA[1]
    factorB = factorttrB[0]
    ttrB = factorttrB[1]
    #shapeA = (factorA.shape[0]*ttrA,factorA.shape[1],factorA.shape[2]*ttrA)
    #shapeB = (factorB.shape[0]*ttrB,factorB.shape[1],factorB.shape[2]*ttrB)
    ret = tl.tensor(np.zeros((factorA.shape[0]*ttrA,factorA.shape[1],factorB.shape[1],factorB.shape[2]*ttrB)))
    for i1 in range(ret.shape[0]):
        for i2 in range(ret.shape[1]):
            for i3 in range(ret.shape[2]):
                for i4 in range(ret.shape[3]):
                    ea = np.array([1 for i in range(ttrA)])
                    eb = np.array([1 for i in range(ttrB)])
                    ret[i1,i2,i3,i4] = np.dot(np.kron(factorA[int(i1/ttrA),i2,:],ea),np.kron(factorB[:,i3,int(i4/ttrB)],eb))
    return ret

def ret_tensor_TTD(factorList):
    ret = factorList[0]
    for i in range(1,len(factorList)):
        print(len(factorList[i]))
        ret = tt_contract(ret, factorList[i])
        print("ret:",ret.shape)
    shp = np.array(ret.shape)
    return ret.reshape(tuple(shp[1:-1]))"""


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
        print("ttr:",tt.rank)
    t6 = tt_join(ttList,toList,corList)
    t7 = tt_join_opt(ttList,toList,corList)
    fit = tm.fit(t6.to_tensor(),t7.to_tensor())
    print(fit)
    

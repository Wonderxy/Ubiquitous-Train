import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
import time
from threading import Thread
from multiprocessing import Process,Pool
from utils.forList import factorial_list
import tensor.base as tb
import copy
import os


def join_order(tensor,order,toList,corList):
    """Obtain the [tensor, order] that joins with [tensor, order]

    Parameters
    ----------
    tensor : int, tensor_id in tList
    order : int, order_id in tensor
    toList : list[list], tensor corresponding to join together
    corList : list, Tensor corresponding to join

    Returns
    -------
    OrderList : list[[tensor,order],[tensor,order],...]
        The [tensor, order] that joins with [tensor, order]
    """
    OrderList = []
    for i in range(1,len(toList)):
        if tensor == toList[i]:
            for j in range(len(corList[i][0])):
                if corList[i][1][j] == order:
                    OrderList.append([i,corList[i][0][j]])
                    break
    for list in OrderList:
        OrderList += join_order(list[0],list[1],toList,corList)
    return OrderList


def Cor_multiply(tList,toList,corList,shpList,FinalOrderList,joinVector,range_):
    joinTensorShape = []
    for site in FinalOrderList:
        joinTensorShape.append(shpList[site[0]][site[1]])
    
    for index in range(range_[0],range_[1]):
        indexList = tb.index_v2t(joinTensorShape,index)
        indexLists = [['' for j in range(len(tList[i].shape))] for i in range(len(tList))]
        for i in range(len(indexList)):
            indexLists[FinalOrderList[i][0]][FinalOrderList[i][1]] = indexList[i]
            JoinOrderList = join_order(FinalOrderList[i][0],FinalOrderList[i][1],toList,corList)
            for list in JoinOrderList:
                indexLists[list[0]][list[1]] = indexList[i]
        result = 1
        for i in range(len(indexLists)):
            vIndex = tb.index_t2v(shpList[i],indexLists[i])
            vector = tb.tensor_to_vec(tList[i])
            result *= vector[vIndex]
        joinVector[index] = result
    return joinVector

class TJMProcess(Process):

    def __init__(self,tList,toList,corList,shpList,FinalOrderList,joinVector,range_):
        Process.__init__(self)
        self.tList = tList
        self.toList = toList
        self.corList = corList
        self.shpList = shpList
        self.FinalOrderList = FinalOrderList
        self.joinVector = copy.deepcopy(joinVector)
        self.range_ = range_

    def start(self):
        self.result = Cor_multiply(self.tList,self.toList,self.corList,self.shpList,\
                                   self.FinalOrderList,self.joinVector,self.range_)


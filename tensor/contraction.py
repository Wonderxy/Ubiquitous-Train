import tensorly as  tl
import numpy as np


def main_join(shpList,corList):
    """Tensor based multi tensor join operation

    Parameters
    ----------
    shpList : list[(tensorshape),(tensorshape),...],tensor shape list
    corList : list[[],[(join_orders),(join_orders_main)],[(),()],...],The relationship between corresponding orders

    Returns
    -------
    FinalOrderList : list[(tensor,order),(tensor,order),...]
        The final positions of the tensors and their orders
    """
    tNum = len(shpList)
    # initialize FinalOrderList
    FinalOrderList = []
    for i in range(len(shpList[0])):
        FinalOrderList.append((0,i))
    
    for i in range(1,tNum):
        # front part
        if corList[i][1][0] > corList[i][0][0]:
            for k in range(corList[i][0][0]):
                insertInedx = FinalOrderList.index((0,corList[i][1][0]))
                FinalOrderList.insert(insertInedx,(i,k))
        else:
            for k in range(corList[i][0][0]):
                FinalOrderList.insert(0+k,(i,k))

        # rear part
        rearNumMain = len(shpList[0])-corList[i][1][-1]
        rearNum = len(shpList[i])-corList[i][0][-1]
        if rearNumMain > rearNum:
            for k in range(corList[i][0][-1]+1,len(shpList[i])):
                insertInedx = FinalOrderList.index((0,corList[i][1][-1]+1))
                FinalOrderList.insert(insertInedx,(i,k))
        else:
            for k in range(corList[i][0][-1]+1,len(shpList[i])):
                FinalOrderList.append((i,k))

        #middle part
        for j in range(len(corList[i][1])-1):
            for k in range(corList[i][0][j]+1,corList[i][0][j+1]):
                insertInedx = FinalOrderList.index((0,corList[i][1][j+1]))
                FinalOrderList.insert(insertInedx,(i,k))

    return FinalOrderList



def tensor_join(tList,toList,corList):
    """Tensor based multi tensor join operation

    Parameters
    ----------
    tList : ndarray/tl.tensor[ndarray/tl.tensor], tensorList
    rList : list[list], orderList
    corList : list, Tensor corresponding to join

    Returns
    -------
    ndarray/tl.tensor
        The Result of Joining Multiple Tensors
    """
    tNum = len(tList)
    if tNum != len(toList) or tNum != len(corList):
        raise ValueError("The number of tensors for joining should be the same, \
                         including tensorList,toList and corList")
    

if __name__ == "__main__":
    t = main_join([(1,1,1,1,1,1,1),(1,1,1,1,1,1),(1,1,1,1,1,1)],[[],[(2,4),(1,4)],[(1,3,4),(2,4,5)]])
    print(t)
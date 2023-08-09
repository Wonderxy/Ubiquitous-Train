import tensorly as  tl
import numpy as np
from treelib import Tree,Node
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
import tensor.base as tb
from utils.forList import factorial_list
import copy
import time
from thread_class.t_multithreading import TJMThread
from process_class.t_multiprocess import TJMProcess

def validate_join_tensor(tList,toList,corList):
    """Verify before operation, including the quantity of tensors and
        the problem of whether the corresponding dimensions between corresponding tensors match

    Parameters
    ----------
    tList : list[tensor/ndarray], tensors List
    toList : list[], tensor corresponding to join together
    corList : list[[],[[join_orders],[join_orders_main]],[[],[]],...],The relationship between corresponding orders

    Returns
    -------
    shpList : list[(shape)], shape List
    lenList : list[], length List
    """
    #validate_tensor_num
    tNum = len(tList)
    if len(toList) != tNum or len(corList) != tNum:
        raise ValueError("The number of parameters for tList,toList,corList needs to be equal")
    #validate_join_order
    shpList = []
    for t in tList:
        shpList.append(t.shape)
    
    for i in range(1,tNum):
        toTensor = toList[i]
        for j in range(len(corList[i][0])):
            if shpList[i][corList[i][0][j]] != shpList[toTensor][corList[i][1][j]]:
                raise ValueError("Mismatch in the number of ordered dimensions between corresponding tensors")
    
    lenList = []
    for shp in shpList:
        lenList.append(len(shp))
    
    return shpList,lenList

def create_tree(toList,lenList,corList):
    """According to the corresponding relationship between the tensors 
    during the join operation, the Spanning tree structure

    Parameters
    ----------
    toList : list[], tensor corresponding to join together
    lenList : [length,...] lenList
    corList : list[[],[[join_orders],[join_orders_main]],[[],[]],...],The relationship between corresponding orders

    Returns
    -------
    joinTree : Tree,the relationship between whether each tensor is joined or not.
        stored the orders of the join and the shape of the tensor in the data section
    """
    #Convert tolist to sonList [[son1,son2,...],[],...]
    sonList = []
    for i in range(len(toList)):
        sons = []
        for j in range(1,len(toList)):
            if toList[j] == i:
                sons.append(j)
        sonList.append(sons)

    joinTree = Tree()
    joinTree.create_node(identifier=str(0),data=[lenList[0],copy.deepcopy(corList[0])])
    joinTree.root = str(0)
    for i in range(len(sonList)):
        for j in sonList[i]:
            joinTree.create_node(identifier=str(j),parent=str(i),data=[lenList[j],copy.deepcopy(corList[j])])   
            
    return joinTree


def tree_join(node,tree):
    """On the basis of Tree structure, multi tensor join is completed through recursion

    Parameters
    ----------
    node : Node
    tree : Tree,joinTree
    Returns
    -------
    FinalOrderList : list[[tensor,order],[tensor,order],...]
        The final positions of the tensors and their orders
    """
    FinalOrderList = []
    #If a leaf node
    if node.is_leaf(tree_id=tree.identifier):
        for i in range(node.data[0]):
            FinalOrderList.append([int(node.tag),i])
        return FinalOrderList
     
    childList = tree.children(node.identifier)

    insertList = {}
    for child in childList:
        insertList[child.identifier] = tree_join(child,tree)

    # initialize FinalOrderList
    for i in range(node.data[0]):
        FinalOrderList.append([int(node.tag),i])
    
    childList = tree.children(node.identifier)
    for child in childList:
        # front part
        firstOrderIndex = FinalOrderList.index([int(node.tag),child.data[1][1][0]])
        if firstOrderIndex > child.data[1][0][0]:
            for k in range(child.data[1][0][0]):
                insertIndex = FinalOrderList.index([int(node.tag),child.data[1][1][0]])
                FinalOrderList.insert(insertIndex,insertList[child.identifier][k])
                node.data[0] += 1
                if node.identifier != tree.root:
                    node.data[1][0] = [i+1 if i>=insertIndex else i for i in node.data[1][0]]
        else:
            for k in range(child.data[1][0][0]):
                FinalOrderList.insert(k,insertList[child.identifier][k])
                node.data[0] += 1
                if node.identifier != tree.root:
                    node.data[1][0] = [i+1 for i in node.data[1][0]]

        #middle part
        for j in range(len(child.data[1][1])-1):
            for k in range(child.data[1][0][j]+1,child.data[1][0][j+1]):
                insertIndex = FinalOrderList.index([int(node.tag),child.data[1][1][j+1]])
                FinalOrderList.insert(insertIndex,insertList[child.identifier][k])
                node.data[0] += 1
                if node.identifier != tree.root:
                    node.data[1][0] = [i+1 if i>=insertIndex else i for i in node.data[1][0]]

        # rear part
        endOrderIndex = FinalOrderList.index([int(node.tag),child.data[1][1][-1]])
        rearNumMain = len(FinalOrderList)-endOrderIndex
        rearNum = child.data[0]-child.data[1][0][-1]
        if rearNumMain > rearNum:
            for k in range(child.data[1][0][-1]+1,child.data[0]):
                insertIndex = FinalOrderList.index([int(node.tag),child.data[1][1][-1]])
                FinalOrderList.insert(insertIndex+1+k-(child.data[1][0][-1]+1),insertList[child.identifier][k])
                node.data[0] += 1
                if node.identifier != tree.root:
                    node.data[1][0] = [i+1 if i>=insertIndex+1+k-(child.data[1][0][-1]+1) else i for i in node.data[1][0]]
        else:
            for k in range(child.data[1][0][-1]+1,child.data[0]):
                FinalOrderList.append(insertList[child.identifier][k])
                node.data[0] += 1

    return FinalOrderList

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

def tensor_join(tList,toList,corList):
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
    shpList,lenList = validate_join_tensor(tList,toList,corList)
    join_tree = create_tree(toList,lenList,corList)
    join_tree.show()
    FinalOrderList = tree_join(join_tree.get_node(join_tree.root),join_tree)

    joinTensorShape = []
    for site in FinalOrderList:
        joinTensorShape.append(shpList[site[0]][site[1]])
    joinVector = tl.tensor(np.zeros(factorial_list(joinTensorShape)))

    # num = int(factorial_list(joinTensorShape)/5)

    # t1 = TJMProcess(tList=tList,toList=toList,corList=corList,shpList=shpList,\
    #         FinalOrderList=FinalOrderList,joinVector=joinVector,\
    #         range_=[0,num])
    
    # t2 = TJMProcess(tList=tList,toList=toList,corList=corList,shpList=shpList,\
    #         FinalOrderList=FinalOrderList,joinVector=joinVector,\
    #         range_=[num,2*num]) 
    
    # t3 = TJMProcess(tList=tList,toList=toList,corList=corList,shpList=shpList,\
    #         FinalOrderList=FinalOrderList,joinVector=joinVector,\
    #         range_=[2*num,3*num]) 
    
    # t4 = TJMProcess(tList=tList,toList=toList,corList=corList,shpList=shpList,\
    #         FinalOrderList=FinalOrderList,joinVector=joinVector,\
    #         range_=[3*num,4*num]) 
    
    # t5 = TJMProcess(tList=tList,toList=toList,corList=corList,shpList=shpList,\
    #         FinalOrderList=FinalOrderList,joinVector=joinVector,\
    #         range_=[4*num,factorial_list(joinTensorShape)]) 
    
    # t1.run()
    # t2.run()
    # t3.run()
    # t4.run()
    # t5.run()
    # result1 = t1.result
    # result2 = t2.result
    # result3 = t3.result
    # result4 = t4.result
    # result5 = t5.result

    # joinVector = result1+result2+result3+result4+result5

    for index in range(factorial_list(joinTensorShape)):
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
    
    joinTensor = tb.vec_to_tensor(joinVector,joinTensorShape)

    return joinTensor 
 

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
    time1 = time.time()
    t6 = tensor_join(tList,toList,corList)
    time2 = time.time()
    print("Shape:",t6.shape)
    print("Time:",time2-time1)

    
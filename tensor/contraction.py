import tensorly as  tl
import numpy as np
from treelib import Tree,Node
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
import tensor.base as tb
from utils.forList import factorial_list

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
            if shpList[i][corList[i][0][j]] != shpList[toTensor][corList[toTensor][1][j]]:
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
    # print(sonList)

    joinTree = Tree()
    joinTree.create_node(identifier=str(0),data=[lenList[0],corList[0]])
    joinTree.root = str(0)
    for i in range(len(sonList)):
        for j in sonList[i]:
            joinTree.create_node(identifier=str(j),parent=str(i),data=[lenList[j],corList[j]])   
            
    return joinTree


def tree_join(node,tree):
    """On the basis of Tree structure, multi tensor join is completed through recursion

    Parameters
    ----------
    node : Node
    tree : Tree,joinTree
    Returns
    -------
    FinalOrderList : list[(tensor,order),(tensor,order),...]
        The final positions of the tensors and their orders
    """
    FinalOrderList = []
    #If a leaf node
    if node.is_leaf(tree_id=tree.identifier):
        for i in range(node.data[0]):
            FinalOrderList.append((int(node.tag),i))
        return FinalOrderList
     
    childList = tree.children(node.identifier)

    insertList = {}
    for child in childList:
        insertList[child.identifier] = tree_join(child,tree)

    # initialize FinalOrderList
    for i in range(node.data[0]):
        FinalOrderList.append((int(node.tag),i))
    
    childList = tree.children(node.identifier)
    for child in childList:
        # front part
        firstOrderIndex = FinalOrderList.index((int(node.tag),child.data[1][1][0]))
        if firstOrderIndex > child.data[1][0][0]:
            for k in range(child.data[1][0][0]):
                insertInedx = FinalOrderList.index((int(node.tag),child.data[1][1][0]))
                FinalOrderList.insert(insertInedx,insertList[child.identifier][k])
                node.data[0] += 1
                if node.identifier != tree.root:
                    node.data[1][0] = [i+1 if i>=insertInedx else i for i in node.data[1][0]]
        else:
            for k in range(child.data[1][0][0]):
                FinalOrderList.insert(k,insertList[child.identifier][k])
                node.data[0] += 1
                if node.identifier != tree.root:
                    node.data[1][0] = [i+1 for i in node.data[1][0]]
        
        #middle part
        for j in range(len(child.data[1][1])-1):
            for k in range(child.data[1][0][j]+1,child.data[1][0][j+1]):
                insertInedx = FinalOrderList.index((int(node.tag),child.data[1][1][j+1]))
                FinalOrderList.insert(insertInedx,insertList[child.identifier][k])
                node.data[0] += 1
                if node.identifier != tree.root:
                    node.data[1][0] = [i+1 if i>=insertInedx else i for i in node.data[1][0]]

        # rear part
        endOrderIndex = FinalOrderList.index((int(node.tag),child.data[1][1][-1]))
        rearNumMain = len(FinalOrderList)-endOrderIndex
        rearNum = child.data[0]-child.data[1][0][-1]
        if rearNumMain > rearNum:
            for k in range(child.data[1][0][-1]+1,child.data[0]):
                insertInedx = FinalOrderList.index((int(node.tag),child.data[1][1][-1]))
                FinalOrderList.insert(insertInedx+1+k-(child.data[1][0][-1]+1),insertList[child.identifier][k])
                node.data[0] += 1
                if node.identifier != tree.root:
                    node.data[1][0] = [i+1 if i>=insertInedx+1+k-(child.data[1][0][-1]+1) else i for i in node.data[1][0]]
        else:
            for k in range(child.data[1][0][-1]+1,child.data[0]):
                FinalOrderList.append(insertList[child.identifier][k])
                node.data[0] += 1
    return FinalOrderList


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

    joinTensorLen = len(FinalOrderList)
    joinTensorShape = []
    for site in FinalOrderList:
        joinTensorShape.append(shpList[site[0]][site[1]])
    joinTensor = tl.tensor(np.zeros(joinTensorShape))
    joinVector = tb.tensor_to_vec(joinTensor)
    
    for index in range(factorial_list(joinTensorShape)):
        indexList = tb.index_v2t(joinTensorShape,index)
        
    

    return FinalOrderList

    

if __name__ == "__main__":
    pass
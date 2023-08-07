import tensorly as  tl
import numpy as np
from treelib import Tree,Node

def validate_join_tensor(tList):

    pass

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
                print(node.data[1])
                node.data[0] += 1
                if node.identifier != tree.root:
                    node.data[1][0] = [i+1 if i>=insertInedx else i for i in node.data[1][0]]
        else:
            for k in range(child.data[1][0][0]):
                FinalOrderList.insert(int(node.tag)+k,insertList[child.identifier][k])
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
                    node.data[1][0] = [i+1 if i>=insertInedx else i for i in node.data[1][0] ]

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
                    node.data[1][0] = [i+1 if i>=insertInedx else i for i in node.data[1][0]]
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
    ndarray/tl.tensor
        The Result of Joining Multiple Tensors
    """
    shpList = validate_join_tensor(tList)

    join_tree = create_tree(toList,shpList,corList)

    FinalOrderList = tree_join(join_tree.get_node(join_tree.root),join_tree)

    return FinalOrderList

    

if __name__ == "__main__":
    lenList = [8,7,5,6]
    toList = [0,0,1,0]
    corList = [[],[[3,5],[2,5]],[[1,3],[3,6]],[[1,2],[2,4]]]
    joinTree = create_tree(toList,lenList,corList)
    joinTree.show()
    FinalOrderList = tree_join(joinTree.get_node(joinTree.root),joinTree)
    print("FinalOrderList:",FinalOrderList)
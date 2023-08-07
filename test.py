import tensorly as  tl
import numpy as np
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
import tensor.base as bt

if __name__ == "__main__":
    T = tl.tensor(np.arange(192).reshape(2, 6, 8, 2))
    shp = T.shape
    V = bt.tensor_to_vec(T)
    index = bt.index_t2v(shp,(0,0,0,1))
    print("index:",index)
    indexList = bt.index_v2t(shp,index)
    print(indexList)

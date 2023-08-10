import tensorly as tl
import numpy as np
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
import decomposition.ttd as ttd
import time
import tt.contraction as ttc
import tensor.contraction as tc
import tensor.mathematical as tm


if __name__ == "__main__":
    t1 = tl.tensor(np.random.randint(0,2,(2,2,3,4,5,6)))
    t2 = tl.tensor(np.random.randint(0,2,(2,3,5,6,7)))
    t3 = tl.tensor(np.random.randint(0,2,(2,3,6,5))) 
    t4 = tl.tensor(np.random.randint(0,2,(2,2,5,2)))
    t5 = tl.tensor(np.random.randint(0,2,(2,3,6,2)))
    tList = [t1,t2,t3,t4,t5]
    toList = [0,0,1,0,1]
    corList = [[],[[1,2],[2,4]],[[1,2],[1,3]],[[1,2],[1,4]],[[1,2],[1,3]]]
    # tList = [t1,t2,t3,t4]
    # toList = [0,0,1,0]
    # corList = [[],[[1,2],[2,4]],[[1,2],[1,3]],[[1,2],[1,4]]]
    ttList = []
    for t in tList:
        tt = ttd.TensorTrain(rank=0,method="tt_svd").fit_transform(t) 
        ttList.append(tt)
    print("tt-svd ok")
    time1 = time.time()
    t6 = ttc.tt_join(ttList,toList,corList)
    time2 = time.time()
    print("tt-Time:",time2-time1)
    print("tt-shape:",t6.to_tensor().shape)
    time3 = time.time()
    t7 = tc.tensor_join(tList,toList,corList)
    time4 = time.time()
    print("t-Time:",time4-time3)
    print("t-shapr:",t7.shape)
    
    fit = tm.fit(t6.to_tensor(),t7)
    print("fit=",fit)

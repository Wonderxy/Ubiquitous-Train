import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
from storage.ttt_access import load_tensor
import tensorly as tl
import numpy as np
import decomposition.ttd as ttd
import copy
import time
import tt.contraction as ttc


path = 'D:/Files/VisualStudioCode/TT2.0/Ubiquitous-Train/dataset'

def exp_runningtime(tList,toList,corList,tNumList,orderNumList,dimNumList):
    resultList = {}
    for tNum in tNumList:
        for orderNum in orderNumList:
            corL = copy.deepcopy(corList[:tNum])
            for i in range(1,len(corL)):
                corL[i][0] = corL[i][0][:orderNum]
                corL[i][1] = corL[i][1][:orderNum]
            print(corL)
            toL = copy.deepcopy(toList[:tNum])
            for dimNum in dimNumList:
                tL = copy.deepcopy(tList[:tNum])
                tL[0] = tL[0][:,:,:,:,:dimNum,:]#need improvement
                tL[1] = tL[1][:,:,:dimNum,:]
                time1 = time.time()
                ttL = []
                for t in tL:
                    tt = ttd.TensorTrain(rank=0,method="tt_svd").fit_transform(t) 
                    ttL.append(tt)
                ttResult = ttc.tt_join(ttL,toL,corL)
                time2 = time.time()
                tt_time = time2 - time1
                resultList[f"{tNum}-{orderNum}-{dimNum}"] = tt_time
    return resultList

            





def exp_fit():
    pass

EXP_FUNS = ["runningtime","fit"]

class Experiment():
    def __init__(self, tList, toList, corList, tNumList, orderNumList, dimNumList, result="runningtime"):
        self.tList = tList
        self.toList = toList
        self.corList = corList
        self.tNumList = tNumList
        self.orderNumList = orderNumList
        self.dimNumList = dimNumList
        self.result = result
    
    def experiment(self):
        if self.result == "runningtime":
            exp_fun = exp_runningtime
        elif self.result == "fit":
            exp_fun = exp_fit
        else:
            raise ValueError(
                f"Got method={self.result}. However, the possible choices are {EXP_FUNS} or to pass a callable."
            )
        self.resultList = exp_fun(self.tList, self.toList, self.corList, self.tNumList,\
                                   self.orderNumList, self.dimNumList)
        


if __name__ == "__main__":
    t1 = tl.tensor(np.random.randint(0,2,(2,2,3,4,1000,6),dtype="bool"))
    t2 = tl.tensor(np.random.randint(0,2,(2,3,1000,6),dtype="bool"))
    t3 = tl.tensor(np.random.randint(0,2,(2,3,6,5),dtype="bool")) 
    t4 = tl.tensor(np.random.randint(0,2,(2,2,6,2),dtype="bool"))
    t5 = tl.tensor(np.random.randint(0,2,(2,2,3,2),dtype="bool"))
    tList = [t1,t2,t3,t4,t5]
    toList = [0,0,1,0,1]
    corList = [[],[[2,3],[4,5]],[[1,2],[1,3]],[[1,2],[1,5]],[[1,2],[0,1]]]
    tNumList = [3,4,5]
    orderNumList = [1,2]
    dimNumList = [200,400,600,800,1000]
    exp = Experiment(tList, toList, corList, tNumList, orderNumList, dimNumList, result="runningtime")
    exp.experiment()
    print(exp.resultList)
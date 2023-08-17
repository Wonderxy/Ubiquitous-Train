import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
from storage.ttt_access import load_tensor
import tensorly as tl
import numpy as np
import decomposition.ttd as ttd
import copy
import time
import tt.contraction as ttc
import tensor.contraction as tc
from utils.print2txt import PrintToTxt,mkdir
import storage.ttt_access as st
import tensor.mathematical as tm


path = 'D:/Files/VisualStudioCode/TT2.0/datastorage/'
printPath = "C:/Users/14619/Desktop/print/8.17/"

def exp_runningtime(tList,epsList,toList,corList,tNumList,orderNumList,dimNumList):
    resultDict = {}
    filePath = mkdir(printPath,mode="now")
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
                tL[0] = tL[0][:,:dimNum,:,:,:,:]#Need improvement
                tL[1] = tL[1][:,:dimNum,:,:]

                time3 = time.time()
                tResult = tc.tensor_join(tL,toL,corL)
                time4 = time.time()
                t_time = time4 - time3
                if dimNum == 1000:#Need improvement
                    st.store_tensor([tResult],[f"tResult-{tNum}-{orderNum}-{dimNum}"],path+"t_storage/")

                for eps in epsList:
                    time1 = time.time()
                    ttL = []
                    for t in tL:
                        tt = ttd.TensorTrain(rank=eps,method="tt_svd").fit_transform(t)
                        print("tt.rank:",tt.rank)
                        ttL.append(tt)
                    ttResult = ttc.tt_join(ttL,toL,corL)
                    time2 = time.time()
                    tt_time = time2 - time1
                    if dimNum == 1000:#Need improvement
                        st.store_tt([ttResult],[f"ttResult-{tNum}-{orderNum}-{dimNum}-{eps}"],path+"tt_storage/")
                    PrintToTxt(filePath+"/",f"runningtime-eps={eps}.txt",\
                        f"tNum={tNum}-orderNum={orderNum}-dimNum={dimNum}:\ntt_time:{tt_time}\nt_time:{t_time}\n","a").write_to_txt()
                    resultDict[f"{tNum}-{orderNum}-{dimNum}-{eps}"] = [tt_time,t_time]
    return resultDict


def exp_fit(tList, epsList, toList, corList, tNumList, orderNumList, dimNumList):
    fitDict = {}
    filePath = mkdir(printPath,mode="now")
    for tNum in tNumList:
        for orderNum in orderNumList:
            tList = load_tensor([f"tResult-{tNum}-{orderNum}-1000"],path+"t_storage/")#Need improvement
            for eps in epsList:
                ttList = st.load_tt([f"ttResult-{tNum}-{orderNum}-1000-{eps}"],path+"tt_storage/")#Need improvement
                fit = tm.fit(tList[0],ttList[0].to_tensor())
                print("eps=",eps)
                print(f"tNum={tNum}-orderNum={orderNum}:\nfit:{fit}\n")
                fitDict[f"{tNum}-{orderNum}-{eps}"] = fit
                PrintToTxt(filePath+"/",f"fit-eps={eps}.txt",\
                        f"tNum={tNum}-orderNum={orderNum}:\nfit:{fit}\n","a").write_to_txt()
    return fitDict


EXP_FUNS = ["runningtime","fit"]

class Experiment():
    def __init__(self, tList, epsList, toList, corList, tNumList, orderNumList, dimNumList, result="runningtime"):
        self.tList = tList
        self.epsList = epsList
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
        self.resultList = exp_fun(self.tList, self.epsList, self.toList, self.corList, self.tNumList,\
                                   self.orderNumList, self.dimNumList)
        


if __name__ == "__main__":
    # t1 = tl.tensor(np.random.randint(0,2,(2,1000,2,2,2,2)))
    # t2 = tl.tensor(np.random.randint(0,2,(2,1000,2,2)))
    # t3 = tl.tensor(np.random.randint(0,2,(2,2,2,2))) 
    # t4 = tl.tensor(np.random.randint(0,2,(2,2,2,2)))
    # t5 = tl.tensor(np.random.randint(0,2,(2,2,2,2)))
    # t6 = tl.tensor(np.random.randint(0,2,(2,2,2,2)))
    # tList = [t1,t2,t3,t4,t5,t6]
    # st.store_tensor([t1,t2,t3,t4,t5,t6],["t1","t2","t3","t4","t5","t6"],path+"/t_storage/")
    tList = st.load_tensor(["t1","t2","t3","t4","t5","t6"],path+"t_storage/")
    toList = [0,0,1,0,1,3]
    corList = [[],[[1,2,3],[1,3,5]],[[0,1,3],[0,2,3]],[[1,2,3],[0,2,5]],[[1,2,3],[0,2,3]],[[0,1,3],[1,2,3]]]
    tNumList = [3,4,5,6]
    orderNumList = [2,3]
    dimNumList = [200,400,600,800,1000]
    epsList = [0.5,0.3,0.2,0.1,0]
    # exp = Experiment(tList, epsList, toList, corList, tNumList, orderNumList, dimNumList, result="runningtime")
    # exp.experiment()
    # print(exp.resultList)
    exp = Experiment(tList, epsList, toList, corList, tNumList, orderNumList, dimNumList, result="fit")
    exp.experiment()
    print(exp.resultList)
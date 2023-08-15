import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')



def exp_runningtime(tList,ttList,tNumList,orderNumList,dimNumList):

    pass


def exp_fit():
    pass

EXP_FUNS = ["runningtime","fit"]

class Experiment():
    def __init__(self, tList, ttList, tNumList, orderNumList, dimNumList, result="runningtime"):
        self.tList = tList
        self.ttList = ttList
        self.tNumList = tNumList
        self.orderNumList = orderNumList
        self.dimNumList = dimNumList
        self.result = result
    
    def experiment(self):
        if self.method == "runningtime":
            exp_fun = exp_runningtime
        elif self.method == "fit":
            exp_fun = exp_fit
        else:
            raise ValueError(
                f"Got method={self.method}. However, the possible choices are {EXP_FUNS} or to pass a callable."
            )
        self.resultList = exp_fun()
        
        
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Ubiquitous-Train")))
from utils.plot import Plot

path = "C:/Users/14619/Desktop/print/8.17/11-13-04"

def getDictString(pendString):
    for i in range(len(pendString)):
        if(pendString[i] == ':'):
            StartIndex = i+1
            break
    dataString = pendString[StartIndex:]
    return dataString

def loadData(path):
    with open(path,"r") as f:
        list1 = []#tensorNum-orderNum
        list2 = []#dimNum
        list3 = []#dimNum
        line = f.readline()
        lineNum = 0
        while line:
            dimNum = lineNum%3
            if dimNum == 1:
                list2.append(float(getDictString(line)))
            if dimNum == 2:
                list3.append(float(getDictString(line)))
            if (lineNum+1)%15 == 0:
                list1.append([list3,list2])
                list2 = []
                list3 = []
            lineNum += 1
            line = f.readline()
    return list1


if __name__ == "__main__":
    epsList = [0.5,0.3,0.2,0.1,0]
    xList = [[["200","400","600","800","1000"] for j in range(6)] for i in range(8)]
    yList = loadData(path+"/runningtime-eps=0.5.txt")
    for eps in epsList[1:]:
        yL = loadData(path+f"/runningtime-eps={eps}.txt")
        for i in range(len(yList)):
            yList[i].append(yL[i][1])
    zList = ["t=3 ord=2","t=3 ord=3","t=4 ord=2","t=4 ord=3","t=5 ord=2","t=5 ord=3","t=6 ord=2","t=6 ord=3"]
    plot = Plot(xAxisList=xList,yAxisList=yList,zAxis=zList,layout="24")
    legendList = ["t-join","tt-0.5","tt-0.3","tt-0.2","tt-0.1","tt-0"]
    xLabel = "dimensions"
    yLabel = "runningtime"
    title = "runningtime"
    figureSize = (10,5)
    plot.multiPlot(legendList=legendList,xLabel=xLabel,yLabel=yLabel,title=title,figureSize=figureSize)
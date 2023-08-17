import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Ubiquitous-Train")))
from utils.plot import Plot

path = "C:/Users/14619/Desktop/print/8.17/11-45-58"

def getDictString(pendString):
    for i in range(len(pendString)):
        if(pendString[i] == ':'):
            StartIndex = i+1
            break
    dataString = pendString[StartIndex:]
    return dataString

def loadData(path):
    with open(path,"r") as f:
        list1 = []
        line = f.readline()
        lineNum = 0
        while line:
            dimNum = lineNum%2
            if dimNum == 1:
                list1.append(float(getDictString(line)))
            lineNum += 1
            line = f.readline()
    return [[list1]]


if __name__ == "__main__":
    epsList = [0.5,0.3,0.2,0.1,0]
    xList = [[["3-2","3-3","4-2","4-3","5-2","5-3","6-2","6-3"] for i in range(5)]]
    yList = loadData(path+"/fit-eps=0.5.txt")
    print(yList)
    for eps in epsList[1:]:
        yL = loadData(path+f"/fit-eps={eps}.txt")
        yList[0].append(yL[0][0])
    plot = Plot(xAxisList=xList,yAxisList=yList)
    legendList = ["eps-0.5","eps-0.3","eps-0.2","eps-0.1","eps-0"]
    xLabel = "tensor-order"
    yLabel = "fit"
    title = "fit"
    plot.singlePlot(legendList=legendList,xLabel=xLabel,yLabel=yLabel,title=title)
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import math

def graphSingle(xAxisList,yAxisList,legendList,xLabel,yLabel,title):
        '''
        xAxisLsit:[[[]]]
        yAxisLsit:[[[1,2,3],[4,5,6],...,[legend]],...,[subplot]] -> plot
        legendList:[]

        '''
        if len(xAxisList) != len(yAxisList):
            raise ValueError("xAxisList yAxisList:Inconsistent list length")
        if len(xAxisList) != 1:
             raise ValueError("xAxisList yAxisList:Dimension can only be 1")
        if len(xAxisList[0]) > 9:
             raise ValueError("Can only draw up to 9 lines") 
        formatList = ['rs-','gs-','bs-','cs-','ms-','ys-','ks-','gs--','bs--']
        plt.yscale('linear')
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.grid(True)
        for i in range(len(xAxisList[0])):
            plt.plot(xAxisList[0][i], yAxisList[0][i],formatList[i])
        plt.legend(legendList)
            
def graphMultiple(xAxisList,yAxisList,zAxisList,layout,legendList,xLabel,yLabel,title,figureSize):
        '''
        xAxisLsit:[[[]]]
        yAxisLsit:[[[1,2,3],[4,5,6],...,[legend]],...,[subplot]] -> plot
        zAxisList:[]
        legendList:[]
        '''
        if not(len(xAxisList) == len(yAxisList) and len(xAxisList) == len(zAxisList)) :
            raise ValueError("xAxisList yAxisList zAxisList:Inconsistent list length")
        plt.figure(figsize=figureSize)
        plt.suptitle(title)
        for i in range(len(xAxisList)):#[[[]->legend,[]]->subplot]
            if len(xAxisList[i]) > 7:
                raise ValueError("Can only draw up to 7 lines") 
            if layout == "":
                posStr = str(math.ceil(np.power(len(zAxisList),0.5)))+str(math.ceil(np.power(len(zAxisList),0.5)))+str(i+1)
            else:
                posStr = layout+str(i+1)
            plt.subplot(int(posStr))
            graphSingle([xAxisList[i]],[yAxisList[i]],legendList,xLabel,yLabel,zAxisList[i])
        plt.gca().yaxis.set_minor_formatter(NullFormatter())
        plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=0.42,wspace=0.5)
        plt.tight_layout()

class Plot:
    def __init__(self,xAxisList,yAxisList,zAxis=[],layout=""):
        self.xAxisList = xAxisList
        self.yAxisList = yAxisList
        self.zAxis = zAxis
        self.layout = layout

    def singlePlot(self,legendList,xLabel,yLabel,title):
        graphSingle(self.xAxisList,self.yAxisList,legendList,xLabel,yLabel,title)
        plt.show()

    def multiPlot(self,legendList,xLabel,yLabel,title,figureSize=(6.4,4.8)):
         graphMultiple(self.xAxisList,self.yAxisList,self.zAxis,self.layout,legendList,xLabel,yLabel,title,figureSize)
         plt.show()
    

    
'''
Example
'''
if __name__ == "__main__":
    xAxisList = [[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]
    # [[[subgraph1_line1_x],[subgraph1_line2_x]],[[subgraph2_line1_x],[subgraph2_line2_x]],...]
    yAxisList = [[[2,4,6],[3,6,9]],[[2,4,6],[4,8,12]],[[2,4,6],[4,8,16]]]
    # [[[subgraph1_line1_y],[subgraph1_line2_y]],[[subgraph2_line1_y],[subgraph2_line2_y]],...]
    zAxis = ['1','2','3']#subgraphName
    legendList = ['jtd','jbd']#legend
    xLabel = 'xLabel'#x-axisName
    yLabel = 'yLabel'#y-axisName
    title = 'title'#entireGraphName
    figureSize = (10,10)#figureSize
    
    Plt = Plot(xAxisList,yAxisList,zAxis)
    Plt.multiPlot(legendList,xLabel,yLabel,title,figureSize)


        
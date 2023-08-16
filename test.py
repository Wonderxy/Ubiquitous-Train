import numpy as np
import tensorly as tl
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
import tensor.mathematical as tm
import os
import datetime

def mkdir(path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)           
            print("---  new folder...  ---")
            print("---  OK  ---")
        else:
            print("---  There is this folder!  ---") 

if __name__ == '__main__':
    strtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # print(strtime.split("-"))
    strlist = strtime.split("-")
    
    file = f"C:/Users/14619/Desktop/print/{strlist[1]}-{strlist[2]}"
    mkdir(file)


import numpy as np
import tensorly as tl
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
import tensor.mathematical as tm
import os
import datetime
import storage.ttt_access as st

path = 'D:/Files/VisualStudioCode/TT2.0/datastorage/'

if __name__ == '__main__':
    ttList = st.load_tt([f"ttResult-3-2-1000-0"],path+"tt_storage/")
    tList = st.load_tensor([f"tResult-3-2-1000"],path+"t_storage/")
    fit = tm.fit(ttList[0].to_tensor(),tList)
    print(fit)
    #test

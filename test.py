import numpy as np
import tensorly as tl
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
import tensor.mathematical as tm

if __name__ == '__main__':
    a = [[],[[1,2,3],[2,2,2]],[[1,2,3],[2,2,2]]]
    b = a
    b[1][0] = b[1][0][:1]
    b[1][1] = b[1][1][:1]
    print(a)

    
import numpy as np
import tensorly as tl
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
import tensor.mathematical as tm

if __name__ == '__main__':
    t1 = tl.tensor(np.random.randint(0,2,(5,5)))
    t2 = tl.tensor(np.random.randint(0,2,(4,4)))
    e1 = tl.tensor(np.eye(5))
    e2 = tl.tensor(np.eye(3))
    e3 = tl.tensor(np.eye(15))
    
    a = np.kron(np.kron(t1,t2),e1)
    b = np.kron(np.kron(t1,e1),t2)

    print(tm.fit(a,b))
    
    
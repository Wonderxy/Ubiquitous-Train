import pickle
import sys
sys.path.append('d:\\Files\\VisualStudioCode\\TT2.0\\Ubiquitous-Train')
import tensor.mathematical as tm
import experiments.load_data as ld


if __name__ == "__main__":
    tensorList1 = ld.load_data()
    tensor1 = tensorList1[2]
    tensor2 = ld.load_tensor(["ratingTensor"])
    fit = tm.fit(tensor1,tensor2)
    print(fit)
    
    

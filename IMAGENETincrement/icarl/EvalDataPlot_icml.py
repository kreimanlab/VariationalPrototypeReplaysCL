
import torch
import os
import os.path
import scipy.io
import numpy as np

# rows: task trials
# cols: performance in task numbers
def convertToMat(prec,n_task):
    mat = np.empty((n_task))
    mat[:] = np.nan
    for i, key in enumerate(prec):
        taskprec = prec[key]        
        mat[i] = taskprec[0]*100                     
    
    print('average accu across classes: ')
    print(mat.mean())
    return mat

namebase = 'results_icml/'
namebase1 = '/home/mengmi/Projects/Proj_CL/code/IMAGENETincrement/Continual-Learning-Benchmark-master/outputs/split_IMAGENET_incre_class_VP/'
n_task = 10
n_repeats = 10

## ProtoCLNetFull20R10, ProtoCLNetFull50R10, ProtoCLNetFull50, ProtoCLNetFull20
loadfilenameprefix = 'ICARL-BCE_'
loadfilenamepostfix = '-precision_record.pt'
savefilenameprefix = 'ICARL_'
savefilenamepostfix = '-precision_record.mat'

for i in range(n_repeats):
    savefilename = os.path.join(namebase1, savefilenameprefix + str(i+1) + savefilenamepostfix)
    loadfilename = os.path.join(namebase, loadfilenameprefix + str(i+1) + loadfilenamepostfix)
    prec = torch.load(loadfilename)
    prec = convertToMat(prec, n_task)
    print(prec)
    print('load from ...')
    print(loadfilename)
    print('save to ...')
    print(savefilename)
    scipy.io.savemat(savefilename, {'prec':prec})




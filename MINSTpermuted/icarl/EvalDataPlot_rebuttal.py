
import torch
import os
import os.path
import scipy.io
import numpy as np

# rows: task trials
# cols: performance in task numbers
def convertToMat(prec,n_task):
    mat = np.empty((n_task,n_task))
    mat[:] = np.nan
    for i, key in enumerate(prec):
        taskprec = prec[key]        
        for tkey in range(len(taskprec)):
            mat[i][tkey] = prec[key][tkey]*100                     
   
    return mat

namebase = 'results_rebuttal/'
namebase1 = '/home/mengmi/Projects/Proj_CL/code/MINSTpermuted/Continual-Learning-Benchmark-master/outputs/permuted_MNIST_incremental_domain_rebuttal/'
n_task = 40
n_repeats = 5

loadfilenameprefix = 'ICARL-BCE_'
loadfilenamepostfix = '-precision_record.pt'
savefilenameprefix = 'ICARL_rebuttal_'
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

loadfilenameprefix = 'ICARL-MSE_'
loadfilenamepostfix = '-precision_record.pt'
savefilenameprefix = 'FSR_rebuttal_'
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
    
    
loadfilenameprefix = 'ICARL-KLD_'
loadfilenamepostfix = '-precision_record.pt'
savefilenameprefix = 'FSRdistill_rebuttal_'
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


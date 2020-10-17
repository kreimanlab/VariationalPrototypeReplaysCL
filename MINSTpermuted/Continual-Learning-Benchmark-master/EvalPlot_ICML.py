#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:21:01 2020

@author: mengmi
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:00:10 2019

@author: mengmi
"""

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
        counter = 0
        for tkey in taskprec:
            mat[i+counter][i] = prec[key][tkey]
            counter = counter + 1           
   
    return mat

namebase = 'outputs/permuted_MNIST_incre_domain/'
n_task = 50

n_repeats = 10

## EWC
loadfilenameprefix = 'EWC_'
loadfilenamepostfix = '-precision_record.pt'

for i in range(n_repeats):
    loadfilename = loadfilenameprefix + str(i+1) + loadfilenamepostfix
    savefilename = loadfilename[:-3] + '.mat'
    savefilename = os.path.join(namebase, savefilename)
    loadfilename = os.path.join(namebase, loadfilename)
    prec = torch.load(loadfilename)
    print(prec)
    prec = convertToMat(prec, n_task)
    print(prec)
    #print(prec)
    print('load from ...')
    print(loadfilename)
    print('save to ...')
    print(savefilename)
    scipy.io.savemat(savefilename, {'prec':prec})
    
## online EWC    
loadfilenameprefix = 'EWC_online_'
loadfilenamepostfix = '-precision_record.pt'

for i in range(n_repeats):
    loadfilename = loadfilenameprefix + str(i+1) + loadfilenamepostfix
    savefilename = loadfilename[:-3] + '.mat'
    savefilename = os.path.join(namebase, savefilename)
    loadfilename = os.path.join(namebase, loadfilename)
    prec = torch.load(loadfilename)
    prec = convertToMat(prec, n_task)
    #print(prec)
    print('load from ...')
    print(loadfilename)
    print('save to ...')
    print(savefilename)
    scipy.io.savemat(savefilename, {'prec':prec})

## NormalNN: chance baseline
loadfilenameprefix = 'NormalNN_'
loadfilenamepostfix = '-precision_record.pt'

for i in range(n_repeats):
    loadfilename = loadfilenameprefix + str(i+1) + loadfilenamepostfix
    savefilename = loadfilename[:-3] + '.mat'
    savefilename = os.path.join(namebase, savefilename)
    loadfilename = os.path.join(namebase, loadfilename)
    prec = torch.load(loadfilename)
    prec = convertToMat(prec, n_task)
    #print(prec)
    print('load from ...')
    print(loadfilename)
    print('save to ...')
    print(savefilename)
    scipy.io.savemat(savefilename, {'prec':prec})

## L2
loadfilenameprefix = 'L2_'
loadfilenamepostfix = '-precision_record.pt'

for i in range(n_repeats):
    loadfilename = loadfilenameprefix + str(i+1) + loadfilenamepostfix
    savefilename = loadfilename[:-3] + '.mat'
    savefilename = os.path.join(namebase, savefilename)
    loadfilename = os.path.join(namebase, loadfilename)
    prec = torch.load(loadfilename)
    prec = convertToMat(prec, n_task)
    #print(prec)
    print('load from ...')
    print(loadfilename)
    print('save to ...')
    print(savefilename)
    scipy.io.savemat(savefilename, {'prec':prec})

## MAS
loadfilenameprefix = 'MAS_'
loadfilenamepostfix = '-precision_record.pt'

for i in range(n_repeats):
    loadfilename = loadfilenameprefix + str(i+1) + loadfilenamepostfix
    savefilename = loadfilename[:-3] + '.mat'
    savefilename = os.path.join(namebase, savefilename)
    loadfilename = os.path.join(namebase, loadfilename)
    prec = torch.load(loadfilename)
    prec = convertToMat(prec, n_task)
    #print(prec)
    print('load from ...')
    print(loadfilename)
    print('save to ...')
    print(savefilename)
    scipy.io.savemat(savefilename, {'prec':prec})

## SI
loadfilenameprefix = 'SI_'
loadfilenamepostfix = '-precision_record.pt'

for i in range(n_repeats):
    loadfilename = loadfilenameprefix + str(i+1) + loadfilenamepostfix
    savefilename = loadfilename[:-3] + '.mat'
    savefilename = os.path.join(namebase, savefilename)
    loadfilename = os.path.join(namebase, loadfilename)
    prec = torch.load(loadfilename)
    prec = convertToMat(prec, n_task)
    #print(prec)
    print('load from ...')
    print(loadfilename)
    print('save to ...')
    print(savefilename)
    scipy.io.savemat(savefilename, {'prec':prec})

















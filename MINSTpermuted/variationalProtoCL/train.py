from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import os
import os.path
import random
import math
import numpy as np
import scipy.io

from data import BalancedDatasetSamplerTrain, BalancedDatasetSamplerTest
from utils import validate, save_checkpoint, load_checkpoint, storeProto, storeOriProto, test


def train(model, train_datasets, test_datasets, args, cuda=False):
    
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #for current task
    #optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #for previous task
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr) #for current task
    optimizer2 = optim.SGD(model.parameters(), lr=args.lr) #for previous task
    optimizer3 = optim.SGD(model.parameters(), lr=args.lr) #for current task
    optimizer4 = optim.SGD(model.parameters(), lr=args.lr) #for current task
    
    model.train()
    
    precision_record = defaultdict(list)
    
    protos_mu = {}
    protos_logvar = {}
    oriprotos = {}

    lossinfor2 = {'loss': float('NaN'), 'acc': float('NaN')}
        
    for task, train_dataset in enumerate(train_datasets, 1):
        best_precision = 0 #the best precision for the current task
        
        if task > 1:
            args.dataset_episodes = args.dataset_nextepisodes
            args.epochs_per_task =  args.epochs_per_tasknext      
        
        for epoch in range(1, args.epochs_per_task+1):
            
            iternum = 0
            
            sampler = BalancedDatasetSamplerTrain(train_dataset,args.dataset_samples,args.dataset_classes,args.dataset_episodes)  
            
            if cuda:
                loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0,pin_memory=True)
            else:
                loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0)
            
            data_stream = tqdm(enumerate(loader, 1))
            for batch_index, (imgs, labels) in data_stream:
                #print('imgs')
                #print(imgs.size())
                # prepare the data.
                imgs = imgs.view(args.dataset_classes, args.dataset_samples, args.dataset_channels, args.dataset_width, args.dataset_width)
                imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
                
                
                # run the model and backpropagate the errors based on current task + previous protos
                optimizer.zero_grad()
                loss, lossinfor = model.loss_initial(imgs, args.dataset_samples,
                                                         args.dataset_nsupport, 
                                                         args.dataset_nquery, 
                                                         args.dataset_classes, 
                                                         args.dataset_channels, 
                                                         args.dataset_width,
                                                         args.temperature)#,
                                                         #protos, cuda) 
                loss.backward()
                optimizer.step()
                
                if task>1:
                    optimizer3.zero_grad()
                    sampleprotos_mu = protos_mu[0]
                    sampleprotos_logvar = protos_logvar[0]
                    rn = random.randint(0,args.dataset_samples-1)
                    imgs_sub = imgs[:,rn,:,:,:].view(args.dataset_classes,1,args.dataset_channels, args.dataset_width, args.dataset_width)
                    loss, lossinfor = model.loss_proto(imgs_sub, sampleprotos_mu, sampleprotos_logvar,
                                                                 args.dataset_samples,
                                                                 args.dataset_nsupport, 
                                                                 args.dataset_nquery, 
                                                                 args.dataset_classes, 
                                                                 args.dataset_channels, 
                                                                 args.dataset_width, 
                                                                 cuda,
                                                                 args.temperature) 
                    loss.backward()
                    optimizer3.step()
                
                
                if iternum%args.replay_freq == 0:
                    # run the model and backpropgate the errors based on previous protos
                    for t in range(task-1):
                        optimizer2.zero_grad()
                        rn = random.randint(0,args.oriproto_eachsize-1)
                        #print('rn:')
                        #print(rn)
                        #print('oriprotos')
                        #print(oriprotos.size())
                        sampleimgs = oriprotos[t][:,rn,:].view(args.dataset_classes,1,args.dataset_channels, args.dataset_width, args.dataset_width)
                        sampleprotos_mu = protos_mu[t]
                        sampleprotos_logvar = protos_logvar[t]
                        loss2, lossinfor2 = model.loss_proto(sampleimgs, sampleprotos_mu, sampleprotos_logvar,
                                                                 args.dataset_samples,
                                                                 args.dataset_nsupport, 
                                                                 args.dataset_nquery, 
                                                                 args.dataset_classes, 
                                                                 args.dataset_channels, 
                                                                 args.dataset_width, 
                                                                 cuda,
                                                                 args.temperature) 
                        loss2.backward()
                        optimizer2.step()
                        
                        optimizer4.zero_grad()
                        rn = random.randint(0,args.oriproto_eachsize-1)
                        #print('rn:')
                        #print(rn)
                        #print('oriprotos')
                        #print(oriprotos.size())
                        sampleimgs = oriprotos[t][:,rn,:].view(args.dataset_classes,1,args.dataset_channels, args.dataset_width, args.dataset_width)
                        sampleprotos_mu = protos_mu[0]
                        sampleprotos_logvar = protos_logvar[0]
                        loss2, lossinfor2 = model.loss_proto(sampleimgs, sampleprotos_mu, sampleprotos_logvar,
                                                                 args.dataset_samples,
                                                                 args.dataset_nsupport, 
                                                                 args.dataset_nquery, 
                                                                 args.dataset_classes, 
                                                                 args.dataset_channels, 
                                                                 args.dataset_width, 
                                                                 cuda,
                                                                 args.temperature) 
                        loss2.backward()
                        optimizer4.step()
                        
                iternum = iternum + 1       
                #print("loss: "+"{}".format(lossinfor['loss'])+"; accuracy: "+ "{}".format(lossinfor['acc']))
                #print(imgs.size());x1 = imgs[1][1][0].view(28,28,1); plt.imshow(x1[:,:,0]); break
        
                data_stream.set_description((
                    'task: {task}/{tasks} | '
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'prec: {prec:.4} | '
                    'loss: {loss:.4} | '
                    'prev_prec: {prevprec:.4} | '
                    'prev_loss: {prevloss:.4} | '
                ).format(
                    task=task,
                    tasks=len(train_datasets),
                    epoch=epoch,
                    epochs=args.epochs_per_task,
                    trained=batch_index*args.batch_size,
                    total=len(loader.dataset),
                    progress=(100.*batch_index*args.batch_size/len(loader.dataset)),
                    prec=lossinfor['acc'],                    
                    loss=lossinfor['loss'],
                    prevprec=lossinfor2['acc'],                    
                    prevloss=lossinfor2['loss'],
                ))
                
            #end of epoch; test in validation set; save best model for this task            
            current_precision = validate(model, test_datasets[task-1], args, cuda)
            print('validation and current_precision: ')
            print(current_precision)
            if current_precision > best_precision:
                best_precision = current_precision
                save_checkpoint(model, args, epoch, current_precision, task)
            
        #end of task;  evaluate on current task and previous tasks
        load_checkpoint(model, args, task, cuda)
        #current_proto, current_oriproto = storeOriProto_Proto(model, train_dataset, args, cuda)
        
        current_proto_mu, current_proto_logvar = storeProto(model, train_dataset, args, cuda)   
        #size: number_task*number_class, num_hidden_unit_dim
        protos_mu[task-1] = current_proto_mu 
        protos_logvar[task-1] = current_proto_logvar
        #print('protos_mu')
        #print(current_proto_mu.size())
        savefilename = args.result_dir + 'protos_repeats_' + str(args.n_repeats) + '_task_' + str(task) + '.mat'
        scipy.io.savemat(savefilename, {'protosmu':current_proto_mu.detach().cpu().numpy(),'protos_logvar':current_proto_logvar.detach().cpu().numpy()})
        
        #args.oriproto_eachsize = int(math.floor(args.oriproto_size/task)) #floor the number of oriprotos per task
        
        current_oriproto = storeOriProto(model, train_dataset, args, cuda) 
        #size: number_task, number_class, number_samples, number_channels, number_width, number_width
        oriprotos[task-1] = current_oriproto
        
        #print('oriprotos')
        #print(current_oriproto.size())
        
        for i in range(len(train_datasets)):
            if i+1<= task:
                #print('shrinking oriprotos') 
                #print(args.oriproto_eachsize)
                #print(oriprotos[i].size())
                oriprotos[i] = oriprotos[i][:,:args.oriproto_eachsize,:]
                #print(oriprotos[i].size())
                #if task == 1:
                    #avg_protos = protos
                    #print('avg_protos size')
                    #print(avg_protos.size())
                #else:
                    #avg_protos = protos[i*args.dataset_classes:(i+1)*args.dataset_classes,:]
                    #for p in range(i+1,task):
                        #avg_protos = avg_protos + protos[args.dataset_classes*p:args.dataset_classes*(p+1),:]
                        #avg_protos = avg_protos/2
                        #print('avg_protos size')
                        #print(avg_protos.size())
                sampleprotos_mu = protos_mu[i]
                sampleprotos_logvar = protos_logvar[i]
                prec = test(model, test_datasets[i], sampleprotos_mu, sampleprotos_logvar, args, cuda)      
                #prec = validateProtos(model, test_datasets[i], args, protos[i*args.dataset_classes:(i+1)*args.dataset_classes,:], cuda)                
                precision_record[task-1].append(prec)
                print('Testing results: (task = ' + str(i+1) + ') at task = ' + str(task) )
                print(precision_record[task-1])
              
        #save precision_record as tensors for plotting later
        path = os.path.join(args.result_dir, '{firstname}_{secondname}-precision_record.pt'.format(firstname=args.model_name,secondname=args.n_repeats))
        torch.save(precision_record, path)        
                
       # print(precision_record)        
                
                
                
                
                
                
                
                
                
                
                
                
                

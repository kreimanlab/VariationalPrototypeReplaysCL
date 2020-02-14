from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import torch
from collections import defaultdict
import os
import os.path
import random
import math
import numpy as np
import scipy.io

from data import BalancedDatasetSampler
from utils import validate, save_checkpoint, load_checkpoint, storeProto, storeOriProto, test


def train(model, train_datasets, test_datasets, task_output_space, args, cuda=False):
    
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #for current task
    #optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #for previous task
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr) #for current task
    optimizer2 = optim.SGD(model.parameters(), lr=args.lr) #for previous task
    
    model.train()
    
    precision_record = defaultdict(list)
        
    protos_mu = {}
    protos_logvar = {}
    oriprotos = {}
    
    lossinfor2 = {'loss': float('NaN'), 'acc': float('NaN')}    
    
    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:',task_names)
    test_dataset_all = []
    train_dataset_all = []
    numclass_all = []

    for task, train_dataset in enumerate(train_datasets, 1):        
        train_name = task_names[task-1]
        train_dataset = train_datasets[train_name]
        test_dataset = test_datasets[train_name]
        train_dataset_all.append(train_dataset)
        test_dataset_all.append(test_dataset)
                
    for task, train_dataset in enumerate(train_datasets, 1):
        best_precision = 0 #the best precision for the current task
         
        if task == 1:
            args.dataset_classes = args.first_split_size
            numclass_all.append(args.dataset_classes)
            args.batch_size = args.first_split_size * args.dataset_samples
            args.dataset_current_classes = args.first_split_size            
            args.dataset_episodes = int(math.floor(args.dataset_total/args.dataset_samples))
        else:
            args.dataset_classes = args.dataset_classes + args.other_split_size
            numclass_all.append(args.dataset_classes)
            args.dataset_samples = args.oriproto_eachsize*2
            args.dataset_episodes = int(math.floor(args.dataset_total/args.dataset_samples))
            args.batch_size = args.other_split_size * args.dataset_samples
            args.dataset_current_classes = args.other_split_size
            
        #print('nsample: ' + str(args.dataset_samples) + '; nepisodes: ' + str(args.dataset_episodes))
        train_name = task_names[task-1]
        train_dataset = train_datasets[train_name]
        
        if task > 1:
            #args.dataset_episodes = args.dataset_nextepisodes
            args.epochs_per_task = args.epochs_per_tasknext
            args.epochs_per_tasknext = int(args.epochs_per_tasknext/2)
            if args.epochs_per_tasknext < 10:
                args.epochs_per_tasknext = 10
        
        for epoch in range(1, args.epochs_per_task+1):
            iternum = 0           
            sampler = BalancedDatasetSampler(train_dataset,args.dataset_samples,args.dataset_episodes)  
   
            if cuda:
                loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0,pin_memory=True)
            else:
                loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0)
            
            data_stream = tqdm(enumerate(loader, 1))
            for batch_index, (imgs, labels, dummy) in data_stream:
                
                # prepare the data.
                imgs = imgs.view(args.dataset_current_classes, args.dataset_samples,args.dataset_channels, args.dataset_width, args.dataset_width)
                imgs = imgs[:, np.random.permutation(np.arange(args.dataset_samples)),:,:,:]
                #print(imgs.size())
                if cuda: imgs = imgs.cuda()  
                
                # combine with previous images
                oldimgs = torch.Tensor()
                for t in range(args.dataset_classes-args.dataset_current_classes):
                    
                    #generate a random index for selecting ori protos for one image per class
                    #rn = random.randint(0,args.oriproto_eachsize-1)
                    #size: num_class, num_sample = 1, num_channle, num_width, num_width
                    samplesingleimgs = oriprotos[t][:,:].view(1,args.oriproto_eachsize,args.dataset_channels, args.dataset_width, args.dataset_width)
                    if cuda: samplesingleimgs = samplesingleimgs.cuda()
                    
                    if oldimgs.nelement() == 0:
                        oldimgs = samplesingleimgs
                    else:
                        oldimgs = torch.cat((oldimgs,samplesingleimgs),0)
                #print('oldimgs')
                #print(oldimgs.size())
                # combine with previous protos
                oldprotos_mu = torch.Tensor()
                oldprotos_logvar = torch.Tensor()
                for t in range(args.dataset_classes-args.dataset_current_classes):
                    #always from the prev task
                    samplesprotos_mu = protos_mu[task-2][t,:]                    
                    samplesprotos_mu = samplesprotos_mu.view(1,args.z_dim)
                    samplesprotos_logvar = protos_logvar[task-2][t,:]
                    samplesprotos_logvar = samplesprotos_logvar.view(1,args.z_dim)
                    if cuda: samplesprotos_mu = samplesprotos_mu.cuda()
                    if cuda: samplesprotos_logvar = samplesprotos_logvar.cuda()
                    
                    if oldprotos_mu.nelement() == 0:
                        oldprotos_mu = samplesprotos_mu
                        oldprotos_logvar = samplesprotos_logvar
                    else:
                        oldprotos_mu = torch.cat((oldprotos_mu,samplesprotos_mu),0)
                        oldprotos_logvar = torch.cat((oldprotos_logvar,samplesprotos_logvar),0)
                        
                if cuda: oldprotos_mu = Variable(oldprotos_mu).cuda()
                if cuda: oldprotos_logvar = Variable(oldprotos_logvar).cuda()
                #print('oldprotos_mu')
                #print(oldprotos_mu.size())
                #print('oldprotos_logvar')
                #print(oldprotos_logvar.size())
                #imgs = torch.cat((oldimgs,imgs),0) 
                #imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
                   
                # run the model and backpropagate the errors based on current task + previous protos
                optimizer.zero_grad()
                #print('model.loss_initial')
                loss, lossinfor = model.loss_initial(imgs, oldimgs, oldprotos_mu, oldprotos_logvar,
                                                         args.dataset_samples,
                                                         args.dataset_current_classes,
                                                         args.dataset_classes,
                                                         args.dataset_nsupport, 
                                                         args.dataset_nquery, 
                                                         args.dataset_channels, 
                                                         args.dataset_width,
                                                         args.temperature)#,
                                                         #protos, cuda) 
                loss.backward()
                optimizer.step() 
                
                if iternum%args.replay_freq == 0:
                    # run the model and backpropgate the errors based on previous protos
                    for T in range(task-1):
                    #if task > 1:
                        
                        # combine with previous images
                        oldimgs = oldimgs.view(args.dataset_classes-args.dataset_current_classes,args.oriproto_eachsize,args.dataset_channels, args.dataset_width, args.dataset_width)
                        #print('oldimgs')
                        #print(oldimgs.size())
                        # combine with previous protos
                        
                        for t in range(task-1):#range(task-2,-1,-1): #range(task-1) in asceding order; range(task-2,-1,-1) in descending order; range(task-2,task-3,-1) in prev task only
                            
                            optimizer2.zero_grad()                            
                            #always from the prev task
                            oldprotos_mu = protos_mu[t][:,:]                     
                            #samplesprotos_mu = samplesprotos_mu.view(1,args.z_dim)
                            oldprotos_logvar = protos_logvar[t][:,:]
                            #alwyas take only one example out of each class and do backprop
                            rn = random.randint(0,args.oriproto_eachsize-1)
                            subsetoldimgs = oldimgs[:numclass_all[t],rn,:].view(numclass_all[t],1,args.dataset_channels, args.dataset_width, args.dataset_width)
                            #print('train recall func - oldprotos_mu')
                            #print(oldprotos_mu.size())
                            #print('train recall func - oldprotos_logvar')
                            #print(oldprotos_logvar.size())
                            #print('train recall func - subsetoldimgs')
                            #print(subsetoldimgs.size())
                            if cuda: subsetoldimgs = Variable(subsetoldimgs).cuda()                            
                            if cuda: oldprotos_mu = Variable(oldprotos_mu).cuda()
                            if cuda: oldprotos_logvar = Variable(oldprotos_logvar).cuda()
                            
                            loss2, lossinfor2 = model.loss_proto(subsetoldimgs, oldprotos_mu, oldprotos_logvar,
                                                             args.dataset_nsupport, 
                                                             args.dataset_nquery,
                                                             numclass_all[t],
                                                             args.dataset_channels, 
                                                             args.dataset_width,
                                                             cuda,
                                                             args.temperature) 
                            loss2.backward()
                            optimizer2.step()
                            
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
                    trained=batch_index*args.batch_size/args.dataset_current_classes,
                    total=args.dataset_episodes* args.dataset_samples,
                    progress=(100.*batch_index*args.batch_size/(args.dataset_current_classes*args.dataset_episodes* args.dataset_samples)),
                    prec=lossinfor['acc'],                    
                    loss=lossinfor['loss'],
                    prevprec=lossinfor2['acc'],                    
                    prevloss=lossinfor2['loss'],
                ))
                
                
                
            
            #end of epoch; test in validation set; save best model for this task 
            val_dataset = torch.utils.data.ConcatDataset(train_dataset_all[:task])  
            #print(len(val_dataset))
            current_precision = validate(model, val_dataset, args, cuda)  
            #print('validation and current_precision: ')
            #print(current_precision)
            #print(best_precision)
            #if current_precision > best_precision:
                #print('saving current model')
            best_precision = current_precision
            #print(best_precision)
            save_checkpoint(model, args, epoch, current_precision, task)
            #else:
                #load_checkpoint(model, args, task, cuda)
                
        #end of task;  evaluate on current task and previous tasks           
            
        load_checkpoint(model, args, task, cuda)
        #current_proto, current_oriproto = storeOriProto_Proto(model, train_dataset, args, cuda)
        
        args.oriproto_eachsize = int(math.floor(args.oriproto_size/args.dataset_classes)) #floor the number of oriprotos per task
        current_oriproto = storeOriProto(model, train_dataset, args, cuda) 
        #print('current_oriproto')
        #print(current_oriproto.size())
        for t in range(args.dataset_current_classes):
            oriprotos[args.dataset_classes - args.dataset_current_classes + t] = current_oriproto[t]
        
        current_proto_mu, current_proto_logvar = storeProto(model, oriprotos, args, cuda)
        current_proto_mu = current_proto_mu.view(args.dataset_classes,args.z_dim) 
        current_proto_logvar = current_proto_logvar.view(args.dataset_classes,args.z_dim)
        protos_mu[task-1] = current_proto_mu
        protos_logvar[task-1] = current_proto_logvar
        #print('train func - protos_mu')
        #print(protos_mu[task-1])
        savefilename = args.result_dir + 'protos_' + str(task) + '.mat'
        scipy.io.savemat(savefilename, {'protosmu':current_proto_mu.detach().cpu().numpy(),'protos_logvar':current_proto_logvar.detach().cpu().numpy()})
        #print('train func - protos_logvar')
        #print(protos_logvar[task-1])
        
        #shrink the storage to make it constant
        for i in range(args.dataset_classes):
                #reduce the storage of previous protos
                oriprotos[i] = oriprotos[i][:args.oriproto_eachsize,:]
                #print('train func - oriprotos[i]:')
                #print(oriprotos[i].size())

        #validate performance in all previous tasks
        test_dataset_con = torch.utils.data.ConcatDataset(test_dataset_all[:task])
        #print(len(val_dataset))      
        prec = test(model, test_dataset_con, protos_mu[task-1], protos_logvar[task-1], args, cuda)      
        precision_record[task-1].append(prec)         
        #print(precision_record)
        print('Testing results: (task = ' + str(task) + ')')
        #print(precision_record[task-1])
        print(precision_record)
        #save precision_record as tensors for plotting later
        path = os.path.join(args.result_dir, '{firstname}_{secondname}-precision_record.pt'.format(firstname=args.model_name,secondname=args.n_repeats))
        torch.save(precision_record, path)        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

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

from data import BalancedDatasetSamplerTrain, BalancedDatasetSamplerTest
from utils import validate, save_checkpoint, load_checkpoint, storeProto, storeOriProto, validateProtos, storeOriProto_Proto, test


def train(model, train_datasets, test_datasets, args, cuda=False):
    
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #for current task
    #optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #for previous task
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr) #for current task
    optimizer2 = optim.SGD(model.parameters(), lr=args.lr) #for previous task
    
    model.train()
    
    precision_record = defaultdict(list)
    
    protos = {}
    oriprotos = {}
    
    lossinfor2 = {'loss': float('NaN'), 'acc': float('NaN')}
        
    for task, train_dataset in enumerate(train_datasets, 1):
        best_precision = 0 #the best precision for the current task
        
        if task > 1:
            args.dataset_episodes = args.dataset_nextepisodes        
            args.epochs_per_task = args.epochs_per_tasknext

        for epoch in range(1, args.epochs_per_task+1):
            
            iternum = 0
            
            sampler = BalancedDatasetSamplerTrain(train_dataset,args.dataset_samples,args.dataset_classes,args.dataset_episodes)  
            
            if cuda:
                loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0,pin_memory=True)
            else:
                loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0)
            
            data_stream = tqdm(enumerate(loader, 1))
            for batch_index, (imgs, labels) in data_stream:
                
                # prepare the data.
                imgs = imgs.view(args.dataset_classes,args.dataset_nquery+args.dataset_nsupport,args.dataset_channels, args.dataset_width, args.dataset_width)
                imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
                
                # run the model and backpropagate the errors based on current task + previous protos
                optimizer.zero_grad()
                loss, lossinfor = model.loss_initial(imgs, args.dataset_nsupport, 
                                                         args.dataset_nquery, 
                                                         args.dataset_classes, 
                                                         args.dataset_channels, 
                                                         args.dataset_width,
                                                         args.temperature)#,
                                                         #protos, cuda) 
                loss.backward()
                optimizer.step()
                
                if iternum%args.replay_freq == 0:
                    # run the model and backpropgate the errors based on previous oriprotos
                    for t in range(task-1):
                        optimizer2.zero_grad()
                        rn = random.randint(0,args.oriproto_eachsize-1)
                        #print('rn:')
                        #print(rn)
                        #print('oriprotos')
                        #print(oriprotos.size())
                        sampleimgs = oriprotos[t][:,rn,:].reshape(args.dataset_classes,args.dataset_nquery+args.dataset_nsupport,args.dataset_channels, args.dataset_width, args.dataset_width)
                        sampleprotos = protos[t][:,rn,:].reshape(args.dataset_classes*(args.dataset_nquery+args.dataset_nsupport),-1)
                        loss2, lossinfor2 = model.loss_proto(sampleimgs, args.dataset_nsupport, 
                                                                 args.dataset_nquery, 
                                                                 args.dataset_classes, 
                                                                 args.dataset_channels, 
                                                                 args.dataset_width,
                                                                 sampleprotos, 
                                                                 cuda,
                                                                 args.temperature,
                                                                 args.model_mode) 
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
                    trained=batch_index*args.batch_size,
                    total=len(loader.dataset),
                    progress=(100.*batch_index*args.batch_size/len(loader.dataset)),
                    prec=lossinfor['acc'],                    
                    loss=lossinfor['loss'],
                    prevprec=lossinfor2['acc'],                    
                    prevloss=lossinfor2['loss'],
                ))
                
            #end of epoch; test in validation set; save best model for this task 
#            print('test_datasets')
#            print(test_datasets[task-1])
            current_precision = validate(model, test_datasets[task-1], args, cuda)
            print('current_precision')
            print(current_precision)
            if current_precision > best_precision:
                best_precision = current_precision
                save_checkpoint(model, args, epoch, current_precision, task)
            
        #end of task;  evaluate on current task and previous tasks
        load_checkpoint(model, args, task, cuda)
        #current_proto, current_oriproto = storeOriProto_Proto(model, train_dataset, args, cuda)
        
        
        
        args.oriproto_eachsize = 1#int(math.floor(args.oriproto_size/task)) #floor the number of oriprotos per task
        current_oriproto = storeOriProto(model, train_dataset, args, cuda) 
        oriprotos[task-1] = current_oriproto
        
        args.proto_eachsize = 10 #int(math.floor(args.proto_size/task))
        current_proto = storeProto(model, train_dataset, args, cuda)        
        protos[task-1] = current_proto 
        
        #print('protos')
        #print(protos.size())
        
        for i in range(len(train_datasets)):
            if i+1<= task:
                #print('shrinking oriprotos') 
                #print(args.oriproto_eachsize)
                #print(oriprotos[i].size())
                oriprotos[i] = oriprotos[i][:,:args.oriproto_eachsize,:]
                protos[i] = protos[i][:,:args.oriproto_eachsize,:]
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
                prec = test(model, test_datasets[i], args, cuda)      
                #prec = validateProtos(model, test_datasets[i], args, protos[i*args.dataset_classes:(i+1)*args.dataset_classes,:], cuda)
                
                precision_record[task-1].append(prec)        
              
        #save precision_record as tensors for plotting later
        path = os.path.join(args.result_dir, '{firstname}_{secondname}-precision_record.pt'.format(firstname=args.model_name,secondname=args.n_repeats))
        torch.save(precision_record, path)        
                
        print(precision_record)        
                
                
                
                
                
                
                
                
                
                
                
                
                

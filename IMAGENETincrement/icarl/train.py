from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import torch
from collections import defaultdict
import os
import os.path
import random
import math

from data import BalancedDatasetSampler
from utils import validate, save_checkpoint, load_checkpoint, storeProto, storeOriProto, test


def train(model, train_datasets, test_datasets, task_output_space, args, cuda=False):
    
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #for current task
    #optimizer2 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #for previous task
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr) #for current task
    optimizer2 = optim.SGD(model.parameters(), lr=args.lr) #for previous task
    
    model.train()
    
    precision_record = defaultdict(list)
        
    protos = {}
    oriprotos = {}
    
    lossinfor2 = {'loss': float('NaN'), 'acc': float('NaN')}
    
    args.dataset_samples = args.dataset_nquery+args.dataset_nsupport
    
    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:',task_names)
    test_dataset_all = []
    train_dataset_all = []
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
            args.batch_size = args.first_split_size * args.dataset_samples
            args.dataset_current_classes = args.first_split_size
        else:
            args.dataset_classes = args.dataset_classes + args.other_split_size
            args.batch_size = args.other_split_size * args.dataset_samples
            args.dataset_current_classes = args.other_split_size
            args.epochs_per_task= args.epochs_per_nexttask
            
        train_name = task_names[task-1]
        train_dataset = train_datasets[train_name]
        
        if task > 1:
            args.dataset_episodes = args.dataset_nextepisodes
        
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
                if cuda: imgs = imgs.cuda()  
                
                # combine with previous images
                oldimgs = torch.Tensor()
                for t in range(args.dataset_classes-args.dataset_current_classes):
                    
                    #generate a random index for selecting ori protos
                    rn = random.randint(0,args.oriproto_eachsize-1)
                    
                    samplesingleimgs = oriprotos[t][rn,:].view(1,args.dataset_samples,args.dataset_channels, args.dataset_width, args.dataset_width)
                    if cuda: samplesingleimgs = samplesingleimgs.cuda()
                    
                    if oldimgs.nelement() == 0:
                        oldimgs = samplesingleimgs
                    else:
                        oldimgs = torch.cat((oldimgs,samplesingleimgs),0)
                        
                if cuda: oldimgs = oldimgs.cuda()        
                imgs = torch.cat((oldimgs,imgs),0) 
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
                
                           
                #every 2 batches; backpropogate original protos
                if iternum%args.replay_freq == 0:
                        # run the model and backpropgate the errors based on previous oriprotos
                        if task > 1:
                            sampleimgs = torch.Tensor()
                            sampleprotos = torch.Tensor()
                            for t in range(args.dataset_classes-args.dataset_current_classes):
                                
                                #generate a random index for selecting ori protos
                                rn = random.randint(0,args.oriproto_eachsize-1)
                                samplesingleimgs = oriprotos[t][rn,:].view(1,args.dataset_samples,args.dataset_channels, args.dataset_width, args.dataset_width)
                                samplesingleprotos = protos[t][rn,:].view(1,args.dataset_samples,-1)
                                                                
                                if cuda: samplesingleimgs = samplesingleimgs.cuda()
                                if cuda: samplesingleprotos = samplesingleprotos.cuda()
                                
                                if sampleimgs.nelement() == 0:
                                    sampleimgs = samplesingleimgs
                                    sampleprotos = samplesingleprotos
                                else:
                                    sampleimgs = torch.cat((sampleimgs,samplesingleimgs),0)
                                    sampleprotos = torch.cat((sampleprotos,samplesingleprotos),0)
            
                            sampleprotos = Variable(sampleprotos).cuda() if cuda else Variable(sampleprotos)
                            sampleimgs = Variable(sampleimgs).cuda() if cuda else Variable(sampleimgs)
                            
                            sampleprotos = sampleprotos.reshape((args.dataset_classes-args.dataset_current_classes)*(args.dataset_samples),-1)
#                            print('sampleimgs')
#                            print(sampleimgs.size())
#                            print('sampleprotos')
#                            print(sampleprotos.size())
                                                        
                            optimizer2.zero_grad()                        
                            loss2, lossinfor2 = model.loss_proto(sampleimgs, args.dataset_nsupport, 
                                                                     args.dataset_nquery, 
                                                                     args.dataset_classes-args.dataset_current_classes, 
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
                    total=args.dataset_episodes* args.dataset_samples,
                    progress=(100.*batch_index*args.batch_size/len(loader.dataset)),
                    prec=lossinfor['acc'],                    
                    loss=lossinfor['loss'],
                    prevprec=lossinfor2['acc'],                    
                    prevloss=lossinfor2['loss'],
                ))
                
            
            #end of epoch; test in validation set; save best model for this task 
            val_dataset = torch.utils.data.ConcatDataset(train_dataset_all[:task])        
            current_precision = validate(model, val_dataset, args, cuda)  
            
            if current_precision > best_precision:
                best_precision = current_precision
                save_checkpoint(model, args, epoch, current_precision, task)
            
        #end of task;  evaluate on current task and previous tasks
        load_checkpoint(model, args, task, cuda)
        #current_proto, current_oriproto = storeOriProto_Proto(model, train_dataset, args, cuda)
        
        args.oriproto_eachsize = int(math.floor(args.oriproto_size/args.dataset_classes)) #floor the number of oriprotos per task
        current_oriproto = storeOriProto(model, train_dataset, args, cuda) 
        args.proto_eachsize = int(math.floor(args.proto_size/args.dataset_classes))
        current_proto = storeProto(model, train_dataset, args, cuda)        
#        print('current_proto')
#        print(current_proto.size())
        
        for t in range(args.dataset_current_classes):
            oriprotos[args.dataset_classes - args.dataset_current_classes + t] = current_oriproto[t]
            protos[args.dataset_classes - args.dataset_current_classes + t] = current_proto[t]
        
        #print('protos')
        #print(protos.size())
        
        for i in range(len(train_datasets)):
            if i+1<= task:
                #reduce the storage of previous protos
                oriprotos[i] = oriprotos[i][:args.oriproto_eachsize,:]
                protos[i] = protos[i][:args.oriproto_eachsize,:]
#                print('task')
#                print(i)
#                print('reduced proto size')
#                print(protos[i].size())
        #validate performance in all previous tasks
        val_dataset = torch.utils.data.ConcatDataset(test_dataset_all[:task])        
        prec = test(model, val_dataset, args, cuda)      
        precision_record[task-1].append(prec)         
        print(precision_record)
        #save precision_record as tensors for plotting later
        path = os.path.join(args.result_dir, '{firstname}_{secondname}-precision_record.pt'.format(firstname=args.model_name,secondname=args.n_repeats))
        torch.save(precision_record, path)      
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

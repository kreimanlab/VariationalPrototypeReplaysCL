import os
import os.path
import torch
from torch.autograd import Variable
from tqdm import tqdm
from data import BalancedDatasetSampler

def test(model, test_dataset, sampleprotos_mu, sampleprotos_logvar, args, cuda=False):
    
    model.train(mode = False) #reset to test mode
    
    sampler = BalancedDatasetSampler(test_dataset,args.dataset_samples,args.test_size)   
            
    if cuda:
        loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, sampler=sampler, batch_size=args.dataset_classes*args.dataset_samples, num_workers=0,pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, sampler=sampler, batch_size=args.dataset_classes*args.dataset_samples, num_workers=0)
            
            
    total_tested = 0
    total_correct = 0
    with torch.no_grad():
        data_stream = tqdm(enumerate(loader, 1))
        for batch_index, (imgs, labels, dummy) in data_stream:
            # break on test size.
            if total_tested >= args.test_size:
                break
            # test the model.
            # prepare the data.
            #print(imgs.size())
            imgs = imgs.view(args.dataset_classes,args.dataset_samples,args.dataset_channels, args.dataset_width, args.dataset_width)
            imgs = imgs[:,0,:,:,:] #take the first sample to test
            imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
                    
            loss, lossinfor = model.loss_proto(imgs, sampleprotos_mu, sampleprotos_logvar,
                                                         args.dataset_nsupport, 
                                                         args.dataset_nquery, 
                                                         args.dataset_classes, 
                                                         args.dataset_channels, 
                                                         args.dataset_width,
                                                         cuda,
                                                         args.temperature)         
            # update statistics.
            total_correct = total_correct + lossinfor['acc']
            total_tested = total_tested + 1
        
    model.train(mode=True) #reset to training mode
    precision = total_correct / total_tested
    
    return precision

#combine all previous tasks together and test
def validate(model, test_dataset, args, cuda=False):
    
    model.train(mode = False) #reset to test mode
    
    sampler = BalancedDatasetSampler(test_dataset,args.dataset_samples,args.dataset_episodes)  
            
    if cuda:
        loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, sampler=sampler, batch_size=args.dataset_classes*args.dataset_samples, num_workers=0,pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, sampler=sampler, batch_size=args.dataset_classes*args.dataset_samples, num_workers=0)
            
            
    total_tested = 0
    total_correct = 0
    
    with torch.no_grad():
        data_stream = tqdm(enumerate(loader, 1))
        for batch_index, (imgs, labels, dummy) in data_stream:
            # break on test size.
            if total_tested >= args.val_size:
                break
            # test the model.
            # prepare the data.
            
            imgs = imgs.view(args.dataset_classes,args.dataset_samples,args.dataset_channels, args.dataset_width, args.dataset_width)
            imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
            
            loss, lossinfor = model.loss_val(imgs, args.dataset_samples,
                                                 args.dataset_classes,
                                                 args.dataset_nsupport, 
                                                 args.dataset_nquery, 
                                                 args.dataset_channels, 
                                                 args.dataset_width,
                                                 args.temperature)         
            # update statistics.
            total_correct = total_correct + lossinfor['acc']
            total_tested = total_tested + 1
        
    model.train(mode=True) #reset to training mode
    precision = total_correct / total_tested
    
    return precision
    
def storeProto(model, oriprotos, args, cuda=False):
    
    model.train(mode = False) #reset to test mode
             
    total_proto_mu = torch.Tensor()
    total_proto_logvar = torch.Tensor()
    
    with torch.no_grad():        
        for clas in range(args.dataset_classes):
            imgs = oriprotos[clas]
            imgs = imgs.view(1,-1,args.dataset_channels, args.dataset_width, args.dataset_width)
            #print('imgs')
            #print(imgs.size())
            nsamples = imgs.size(1)
            imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
                    
            current_mu, current_logvar = model.getHiddenReps(imgs, nsamples,
                                                args.dataset_nsupport, 
                                                 args.dataset_nquery, 
                                                 1, 
                                                 args.dataset_channels, 
                                                 args.dataset_width) 
            
            # update statistics.
            if total_proto_mu.nelement() == 0:
                total_proto_mu = current_mu.detach().cpu()
                total_proto_logvar = current_logvar.detach().cpu()
            else:
                total_proto_mu = torch.cat((total_proto_mu,current_mu.detach().cpu().clone()),0) 
                total_proto_logvar = torch.cat((total_proto_logvar,current_logvar.detach().cpu().clone()),0) 
                
    #print('total_proto_mu')  
    #print(total_proto_mu.size())
    #print('total_proto_logvar')  
    #print(total_proto_logvar.size())                      
    model.train(mode=True) #reset to training mode
        
    return total_proto_mu, total_proto_logvar
    
def storeOriProto(model, train_dataset, args, cuda=False):
    
    sampler = BalancedDatasetSampler(train_dataset,args.dataset_total,1)        
    if cuda:
        loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.dataset_total*args.dataset_current_classes, num_workers=0,pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.dataset_total*args.dataset_current_classes, num_workers=0)
            
            
    total_proto = torch.Tensor()
    total_tested = 0
    data_stream = tqdm(enumerate(loader, 1))
    for batch_index, (imgs, labels, dummy) in data_stream:
        
        if total_tested >= args.oriproto_eachsize:
            break
        # test the model.
        # prepare the data.
        #print('img1')
        #print(imgs.size())
        imgs = imgs.view(args.dataset_current_classes, args.dataset_total,args.dataset_channels, args.dataset_width, args.dataset_width) 
        #print('img2')
        #print(imgs.size())
        #current = imgs[:,0:args.proto_size-1,:]
        current = imgs
                
        if total_proto.nelement() == 0:
            total_proto = current.detach().cpu().clone()
        else:
            total_proto = torch.cat((total_proto,current.detach().cpu().clone()),1) 
        
        total_tested = total_tested + 1 
        #print('current')
        #print(current.size())
    
    total_proto = total_proto.view(args.dataset_current_classes, args.dataset_total,args.dataset_channels, args.dataset_width, args.dataset_width) 
    return total_proto


def save_checkpoint(model, args, epoch, precision, task):
    
    path = os.path.join(args.model_dir, '{name}-task-{task}-best'.format(name=args.model_name, task=task))

    # save the checkpoint.
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    torch.save({
        'state': model.state_dict(),
        'epoch': epoch,
        'precision': precision,
        'task': task
    }, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(name=args.model_name, path=path))

def load_checkpoint(model, args, task, cuda):
    
    path = os.path.join(args.model_dir, '{name}-task-{task}-best'.format(name=args.model_name, task=task))

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(name=args.model_name, path=path))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    
    if cuda:
        model.cuda()
        
    epoch = checkpoint['epoch']
    precision = checkpoint['precision']
    return epoch, precision






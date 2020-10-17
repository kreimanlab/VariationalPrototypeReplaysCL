import os
import os.path
import torch
from torch.autograd import Variable
from tqdm import tqdm
from data import BalancedDatasetSamplerTrain, BalancedDatasetSamplerTest


def validateProtos(model, test_dataset, args, avg_protos, cuda=False):
    
    model.train(mode = False) #reset to test mode
    
    sampler = BalancedDatasetSamplerTest(test_dataset,args.dataset_samples,args.dataset_classes,args.dataset_episodes)  
            
    if cuda:
        loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0,pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0)
            
            
    total_tested = 0
    total_correct = 0
    with torch.no_grad():
        data_stream = tqdm(enumerate(loader, 1))
        for batch_index, (imgs, labels) in data_stream:
            # break on test size.
            if total_tested >= args.test_size:
                break
            # test the model.
            # prepare the data.
            imgs = imgs.view(args.dataset_classes,args.dataset_nquery+args.dataset_nsupport,args.dataset_channels, args.dataset_width, args.dataset_width)
            imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
                    
            loss, lossinfor = model.validate_protos(imgs,args.dataset_nsupport, 
                                                 args.dataset_nquery, 
                                                 args.dataset_classes, 
                                                 args.dataset_channels, 
                                                 args.dataset_width,
                                                 avg_protos,
                                                 args.temperature)         
            # update statistics.
            total_correct = total_correct + lossinfor['acc']
            total_tested = total_tested + 1
        
    model.train(mode=True) #reset to training mode
    precision = total_correct / total_tested
    
    return precision

def validate(model, test_dataset, args, cuda=False):
    
    model.train(mode = False) #reset to test mode
    
    sampler = BalancedDatasetSamplerTest(test_dataset,args.dataset_samples,args.dataset_classes,args.dataset_episodes)  
            
    if cuda:
        loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0,pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0)
            
            
    total_tested = 0
    total_correct = 0
    with torch.no_grad():
        data_stream = tqdm(enumerate(loader, 1))
        for batch_index, (imgs, labels) in data_stream:
            # break on test size.
            if total_tested >= args.test_size:
                break
            # test the model.
            # prepare the data.
            imgs = imgs.view(args.dataset_classes,args.dataset_nquery+args.dataset_nsupport,args.dataset_channels, args.dataset_width, args.dataset_width)
            imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
                    
            loss, lossinfor = model.loss_initial(imgs,args.dataset_nsupport, 
                                                 args.dataset_nquery, 
                                                 args.dataset_classes, 
                                                 args.dataset_channels, 
                                                 args.dataset_width,
                                                 args.temperature)         
            # update statistics.
            total_correct = total_correct + lossinfor['acc']
            total_tested = total_tested + 1
        
    model.train(mode=True) #reset to training mode
    precision = total_correct / total_tested
    
    return precision

def test(model, test_dataset, args, cuda=False):
    
    model.train(mode = False) #reset to test mode
    
    sampler = BalancedDatasetSamplerTest(test_dataset,args.dataset_samples,args.dataset_classes,args.test_size)  
            
    if cuda:
        loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0,pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0)
            
            
    total_tested = 0
    total_correct = 0
    with torch.no_grad():
        data_stream = tqdm(enumerate(loader, 1))
        for batch_index, (imgs, labels) in data_stream:
            # break on test size.
            if total_tested >= args.test_size:
                break
            # test the model.
            # prepare the data.
            imgs = imgs.view(args.dataset_classes,args.dataset_nquery+args.dataset_nsupport,args.dataset_channels, args.dataset_width, args.dataset_width)
            imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
                    
            loss, lossinfor = model.loss_initial(imgs,args.dataset_nsupport, 
                                                 args.dataset_nquery, 
                                                 args.dataset_classes, 
                                                 args.dataset_channels, 
                                                 args.dataset_width,
                                                 args.temperature)         
            # update statistics.
            total_correct = total_correct + lossinfor['acc']
            total_tested = total_tested + 1
        
    model.train(mode=True) #reset to training mode
    precision = total_correct / total_tested
    
    return precision
    
def storeProto(model, train_dataset, args, cuda=False):
    
    model.train(mode = False) #reset to test mode
    
    sampler = BalancedDatasetSamplerTrain(train_dataset,args.dataset_samples,args.dataset_classes,args.dataset_episodes)  
            
    if cuda:
        loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0,pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0)
            
            
    total_proto = torch.Tensor()
    total_tested = 0    
    
    with torch.no_grad():
        data_stream = tqdm(enumerate(loader, 1))
        for batch_index, (imgs, labels) in data_stream:
            # break on test size.
            if total_tested >= args.oriproto_eachsize:
                break
            # test the model.
            # prepare the data.
            imgs = imgs.view(args.dataset_classes,args.dataset_nquery+args.dataset_nsupport,args.dataset_channels, args.dataset_width, args.dataset_width)
            imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
#            print('imgs')
#            print(imgs.size())
            current = model.getHiddenReps(imgs,args.dataset_nsupport, 
                                                 args.dataset_nquery, 
                                                 args.dataset_classes, 
                                                 args.dataset_channels, 
                                                 args.dataset_width) 
            
            
            # update statistics.
            if total_proto.nelement() == 0:
                total_proto = current.detach().cpu().reshape(args.dataset_classes,1,args.dataset_nquery+args.dataset_nsupport,-1)
            else:
                
                current = current.detach().cpu().clone().reshape(args.dataset_classes,1,args.dataset_nquery+args.dataset_nsupport,-1)
#                print('current')
#                print(current.size()) 
#                print('total_proto')
#                print(total_proto.size())
                total_proto = torch.cat((total_proto,current),1)    
                
            total_tested = total_tested + 1         
            
    model.train(mode=True) #reset to training mode
    #total_proto = total_proto/total_tested
    #print(total_proto)
    
    return total_proto


def storeOriProto(model, train_dataset, args, cuda=False):
    
    sampler = BalancedDatasetSamplerTrain(train_dataset,args.dataset_samples,args.dataset_classes,args.dataset_episodes)  
            
    if cuda:
        loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0,pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0)
            
            
    total_proto = torch.Tensor()
    total_tested = 0
    data_stream = tqdm(enumerate(loader, 1))
    for batch_index, (imgs, labels) in data_stream:
        
        if total_tested >= args.oriproto_eachsize:
            break
        # test the model.
        # prepare the data.
        #print('img1')
        #print(imgs.size())
        imgs = imgs.view(args.dataset_classes, 1, args.dataset_nquery+args.dataset_nsupport,args.dataset_channels, args.dataset_width, args.dataset_width) 
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
    
    return total_proto


def storeOriProto_Proto(model, train_dataset, args, cuda=False):
    
    model.train(mode = False) #reset to test mode
    
    sampler = BalancedDatasetSamplerTrain(train_dataset,args.dataset_samples,args.dataset_classes,args.dataset_episodes)  
            
    if cuda:
        loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0,pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0)
            
            
    total_proto = torch.Tensor()
    total_tested = 0    
       
    with torch.no_grad():
        data_stream = tqdm(enumerate(loader, 1))
        for batch_index, (imgs, labels) in data_stream:
            
            # prepare the data.
            imgs = imgs.view(args.dataset_classes,args.dataset_nquery+args.dataset_nsupport,args.dataset_channels, args.dataset_width, args.dataset_width)
            imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
                       
            current = model.getHiddenReps(imgs,args.dataset_nsupport, 
                                                 args.dataset_nquery, 
                                                 args.dataset_classes, 
                                                 args.dataset_channels, 
                                                 args.dataset_width) 
            
            # update statistics.
            if total_proto.nelement() == 0:
                total_proto = current.detach().cpu()
            else:
                total_proto = total_proto + current.detach().cpu()   
                
            total_tested = total_tested + 1                
            
    model.train(mode=True) #reset to training mode
    total_proto = total_proto/total_tested
    #print(total_proto)
    
    ################# find best ori_proto #####################
    model.train(mode = False) #reset to test mode
    
    sampler = BalancedDatasetSamplerTrain(train_dataset,args.dataset_samples,args.dataset_classes,args.dataset_episodes)  
            
    if cuda:
        loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0,pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=0)
            
            
    total_ratio = 0    
    with torch.no_grad():
        data_stream = tqdm(enumerate(loader, 1))
        for batch_index, (imgs, labels) in data_stream:
            
            # prepare the data.
            imgs = imgs.view(args.dataset_classes,args.dataset_nquery+args.dataset_nsupport,args.dataset_channels, args.dataset_width, args.dataset_width)
            imgs = Variable(imgs).cuda() if cuda else Variable(imgs)
            
            current_ratio = model.calculateDist_protos(imgs,args.dataset_nsupport, 
                                                 args.dataset_nquery, 
                                                 args.dataset_classes, 
                                                 args.dataset_channels, 
                                                 args.dataset_width,
                                                 total_proto,
                                                 args.temperature)
            if cuda:
                current = imgs.cpu().clone()
            
            if total_ratio < current_ratio:                
                total_ratio = current_ratio            
                best_oriproto = current.clone()
               
        #print('current')
        #print(current.size())
    model.train(mode=True) #reset to training mode
    
    return total_proto, best_oriproto

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






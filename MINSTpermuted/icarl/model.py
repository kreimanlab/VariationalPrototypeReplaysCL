import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict

def euclidean_dist(x, y, temperature):
    # x: N x D; n_class, z_dim
    # y: M x D; n_class*n_query, z_dim; protos
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)/temperature

    # cosine loss
#    n = x.size(0)
#    m = y.size(0)
#    d = x.size(1)
#    assert d == y.size(1)
#    x = x.unsqueeze(1).expand(n, m, d)
#    y = y.unsqueeze(0).expand(n, m, d)
#    return -torch.mul(x,y).sum(2)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
class NormalizeUnitLenL2(nn.Module):
    def __init__(self):
        super(NormalizeUnitLenL2, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)
    
class Protonet(nn.Module):
    def __init__(self, encoder, args):
        super(Protonet, self).__init__()        
        self.encoder = encoder
        self.cls_loss = nn.CrossEntropyLoss()
        
        if args.model_mode == 1:
            self.dist_loss = nn.MSELoss() #nn.MSELoss() #nn.BCELoss() #nn.KLDivLoss()
        elif args.model_mode == 2:
            self.dist_loss = nn.BCELoss()
        else:
            self.dist_loss = nn.KLDivLoss()
        
    def forward(self,sample_inputs):
        z = self.encoder.forward(sample_inputs)
        return z  
        
    def getHiddenReps(self, sample_inputs, n_xs, n_xq, n_class, n_channles, n_size):
                
        sample_inputs = Variable(sample_inputs.reshape(n_class*(n_xs+n_xq), n_channles* n_size* n_size))
        z = self.encoder.forward(sample_inputs) 
        z_dim = z.size(-1)
        z_proto = z.view(n_class, n_xs+n_xq, z_dim)
        return z_proto
        
    def validate_protos(self, sample_inputs, n_xs, n_xq, n_class, n_channles, n_size, avg_protos, temperature):
        xq = Variable(sample_inputs[:,:n_xs+n_xq,:]) # query;         
        n_query = xq.size(1)
        
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()
            avg_protos = avg_protos.cuda()
        
        xq = xq.reshape(n_class * n_query, n_channles* n_size*n_size)        
        
        x = xq
        z = self.encoder.forward(x)        
        z_proto = avg_protos
        zq = z

        #z_proto: n_class, z_dim
        #zq: n_class*n_query, z_dim
        dists = euclidean_dist(zq, z_proto,temperature)

        #dists: n_class*n_query, n_class
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        #log_p_y: n_class, n_query, n_class (normalized from 0 to 1)
        #target_inds: n_class, n_query, 1
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        #pick the values of the ground truth index and calculate cross entropy loss
        _, y_hat = log_p_y.max(2)
        
        #y_hat: [n_class, n_query] ->  index of max value
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }
        
    def calculateDist_protos(self, sample_inputs, n_xs, n_xq, n_class, n_channles, n_size, avg_protos, temperature):
        xq = Variable(sample_inputs[:,:n_xs+n_xq,:]) # query;         
        n_query = xq.size(1)
        
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()
            avg_protos = avg_protos.cuda()
        
        xq = xq.reshape(n_class * n_query, n_channles*n_size* n_size)        
        
        x = xq
        z = self.encoder.forward(x)        
        z_proto = avg_protos
        zq = z

        #z_proto: n_class, z_dim
        #zq: n_class*n_query, z_dim
        dists = euclidean_dist(zq, z_proto,temperature)        
        dists = dists.view(n_class, n_query, n_class)
        
        #we want the dists within classes to be large; the dists between classes to be small
        distratio = torch.Tensor(n_class)
        for cl in range(n_class):
            mat_cl = dists[cl,:]
            mat_cl_same = mat_cl[:,cl].sum()
            mat_cl_diff = mat_cl.sum() - mat_cl[:,cl].sum()
            mat_ratio = mat_cl_same/mat_cl_diff
            distratio[cl] = mat_ratio

        return distratio.mean()
    
    def loss_initial(self, sample_inputs, n_xs, n_xq, n_class, n_channles, n_size, temperature):
        
        target_inds = torch.arange(0, n_class).view(n_class, 1).expand(n_class, n_xs+n_xq).long().reshape(-1)
        target_inds = Variable(target_inds)

        if sample_inputs.is_cuda:
            target_inds = target_inds.cuda()
        
        x = sample_inputs.reshape(n_class*(n_xs + n_xq), n_channles* n_size* n_size)
        
        z = self.encoder.forward(x)        
        loss_val = self.cls_loss(z,target_inds)
 
        #pick the values of the ground truth index and calculate cross entropy loss
        _, y_hat = z.softmax(1).max(1)
        
        #y_hat: [n_class, n_query] ->  index of max value
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }
        
    def loss_proto(self, sample_inputs, n_xs, n_xq, n_class, n_channles, n_size, protos, cuda, temperature, model_mode):
        
        if cuda: sample_inputs = sample_inputs.cuda()
        
        protos = Variable(protos)
                         
        target_inds = torch.arange(0, n_class).view(n_class, 1).expand(n_class, n_xs+n_xq).long().reshape(-1)
        target_inds = Variable(target_inds)
        protos = Variable(protos)

        if sample_inputs.is_cuda:
            target_inds = target_inds.cuda()
            protos = protos.cuda()
            
        x = sample_inputs.reshape(n_class*(n_xs + n_xq), n_channles* n_size* n_size)
        z = self.encoder.forward(x)        
        #loss_val = self.cls_loss(z,target_inds)

        #pick the values of the ground truth index and calculate cross entropy loss
        _, y_hat = z.softmax(1).max(1)
        
        #y_hat: [n_class, n_query] ->  index of max value
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        
        ## we calculate losses for each previous task 
       
        
        
        if model_mode == 1:
            ## MSEloss
            dist_loss = self.dist_loss(z,protos)
        elif model_mode == 2:
             ## BCEloss
            F_z = torch.sigmoid(z)
            F_protos = torch.sigmoid(protos)
            dist_loss = self.dist_loss(F_z, F_protos) 
        else:
            ## KLDloss
            dist_loss = self.dist_loss(z.softmax(1),protos.softmax(1))
        
        
        #loss_total = loss_val + dist_loss
        loss_total = dist_loss
        acc_total = acc_val
                    
        return loss_total, {
            'loss': loss_total.item(),
            'acc': acc_total.item()
        }
        
        
    def loss_oriproto(self, sample_inputs, n_xs, n_xq, n_class, n_channles, n_size, protos, temperature):
            xs = Variable(sample_inputs[:,:n_xs,:]) # support; 
            xq = Variable(sample_inputs[:,n_xs:,:]) # query; 
            protos = Variable(protos)
            
            #print('protos')
            #print(protos.size())
            
            n_class = xs.size(0)
            n_support = xs.size(1)
            n_query = xq.size(1)
            n_proto = protos.size(0)
            n_prevtask = n_proto/n_class #we store one proto for each class for each previous task
            
            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
            target_inds = Variable(target_inds, requires_grad=False)
    
            if xq.is_cuda:
                target_inds = target_inds.cuda()
                protos = protos.cuda()
                
            xs = xs.reshape(n_class*n_support, n_channles* n_size* n_size)
            xq = xq.reshape(n_class * n_query, n_channles* n_size* n_size)        
            
            x = torch.cat((xs,xq), 0)
            z = self.encoder.forward(x)        
            z_dim = z.size(-1)
    
            zq = z[n_class*n_support:]
            z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
            
            #z_proto: n_class, z_dim
            #zq: n_class*n_query, z_dim
            dists = euclidean_dist(zq, z_proto,temperature)
    
            #dists: n_class*n_query, n_class
            log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
    
            #log_p_y: n_class, n_query, n_class (normalized from 0 to 1)
            #target_inds: n_class, n_query, 1
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
            #print(loss_val)
            #pick the values of the ground truth index and calculate cross entropy loss
            _, y_hat = log_p_y.max(2)
            
            #y_hat: [n_class, n_query] ->  index of max value
            acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
            
            ## we calculate losses for each previous task
            z_proto_p = defaultdict(list)
            dists_p = defaultdict(list)
            log_p_y_p = defaultdict(list)
            y_hat_p = defaultdict(list)
            loss_val_p = defaultdict(list)
            acc_val_p = defaultdict(list)
            
            if n_prevtask > 0:
                protos_c = protos.view(-1,n_channles* n_size* n_size)
                z_protos = self.encoder.forward(protos_c)
                z_protos = z_protos.view(n_class*n_prevtask, -1, z_dim).mean(1)
            
            for t in range(n_prevtask): 
                
                z_proto_p[t] = z_protos[t*n_class:(t+1)*n_class,:]
                z_proto_p[t] = z_proto_p[t]
                #print(z_proto_p[t].size())
                dists_p[t] = euclidean_dist(zq, z_proto_p[t],temperature)
                #print(dists_p[t].size())
                log_p_y_p[t] = F.log_softmax(-dists_p[t], dim=1).view(n_class, n_query, -1)
                #print(log_p_y_p[t].size())
                loss_val_p[t] = - log_p_y_p[t].gather(2, target_inds).squeeze().view(-1).mean()
                _, y_hat_p[t] = log_p_y_p[t].max(2)
                acc_val_p[t] = torch.eq(y_hat_p[t], target_inds.squeeze()).float().mean()
                #print('loss_val_p and acc_val_p')
                #print(loss_val_p[t])
                #print(acc_val_p[t])   
            
            loss_total = loss_val
            acc_total = acc_val
            for t in range(n_prevtask):
                loss_total = loss_total + loss_val_p[t]
                acc_total = acc_total + acc_val_p[t]
                
                
            return loss_total, {
                'loss': loss_total.item()/(n_prevtask+1),
                'acc': acc_total.item()/(n_prevtask+1)
            }
            
##########################  END OF MODEL DEFINITION  ##########################
        
        
def load_protonet_conv(args):
    x_dim = args.x_dim
    hid_dim = args.hid_dim
    z_dim = args.z_dim

#    encoder = nn.Sequential(
#    conv_block(x_dim, hid_dim),
#    conv_block(hid_dim, hid_dim),
#    conv_block(hid_dim, hid_dim), #output size: batchsize * 64 * 3 * 3
#    conv_block(hid_dim, z_dim),
#    Flatten()#,
#    #NormalizeUnitLenL2()        
#    )

    encoder = nn.Sequential(
        nn.Linear(in_features=x_dim, out_features=hid_dim, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=hid_dim, out_features=z_dim, bias=True),
        #nn.ReLU(),
        Flatten()#,
        #NormalizeUnitLenL2()        
    )

    return Protonet(encoder,args)
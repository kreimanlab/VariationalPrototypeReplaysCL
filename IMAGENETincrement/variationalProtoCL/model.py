import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict
import math

def euclidean_dist(x, y, temperature):
    # x: N x D; n_class, z_dim
    # y: M x D; n_class*n_query, z_dim; protos
    # L2 loss
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    #w = w.unsqueeze(0).expand(n, m, d)
    temperature = 2
    return torch.pow(x - y, 2).sum(2)/temperature

def euclidean_dist_weighted(x, y, w, temperature):
    # x: N x D; n_class, z_dim
    # y: M x D; n_class*n_query, z_dim; protos
    # L2 loss
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    w = w.unsqueeze(0).expand(n, m, d)
    temperature = 2
    return torch.mul(w, torch.pow(x - y, 2)).sum(2)/temperature

    # cosine loss
#    n = x.size(0)
#    m = y.size(0)
#    d = x.size(1)
#    assert d == y.size(1)
#    x = x.unsqueeze(1).expand(n, m, d)
#    y = y.unsqueeze(0).expand(n, m, d)
#    return -torch.mul(x,y).sum(2)/temperature

    # KLD loss
    # kl2 = (model2 * np.log(model2/actual)).sum()
#    n = x.size(0)
#    m = y.size(0)
#    d = x.size(1)
#    assert d == y.size(1)
#    x = x.unsqueeze(1).expand(n, m, d).softmax(2)
#    y = y.unsqueeze(0).expand(n, m, d).softmax(2)
#    return torch.mul(x, torch.log( torch.div(x,y))).sum(2)/temperature

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
    def __init__(self, args):
        super(Protonet, self).__init__()        
        
        z_dim = args.z_dim
        feat_map_sz = args.dataset_width//16
        n_feat = 256 * feat_map_sz * feat_map_sz #1024
        in_channel = args.dataset_channels
    
        self.enc = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
    
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),        
    
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
    
            Flatten(),
            nn.Linear(n_feat, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(inplace=True),
            
            Flatten()#,
            #nn.Softmax(dim=1)
        )   
        
        self.fc31 = nn.Linear(z_dim,  z_dim)
        self.fc32 = nn.Linear(z_dim,  z_dim)        
        
    def encode(self, x):
        encout = self.enc(x)        
        return self.fc31(encout), self.fc32(encout)
    
    def reparameterize(self, mu, logvar, N):
        
        z = torch.Tensor()
        for i in range(N):
            
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            
            if z.nelement() == 0:
                z = mu + eps*std
                z = torch.unsqueeze(z, 1)
                #print('unsqeeuze')
                #print(z.size())
            else:
                val = mu + eps*std
                val = torch.unsqueeze(val,1)
                z = torch.cat((z,val),1)
            
        return z
    
    def forward(self, x, N):
        mu, logvar = self.encode(x)        
        z = self.reparameterize(mu, logvar, N)
        return z, mu, logvar
        
    def getHiddenReps(self, sample_inputs, n_sam, n_xs, n_xq, n_class, n_channles, n_size):
        sample_inputs = sample_inputs.reshape(n_class*n_sam, n_channles, n_size, n_size)
        #z: n_class, n_samples, n_channles*n_size*n_size
        z, mu, logvar = self.forward(sample_inputs, n_xs+n_xq)
        #sample_inputs = Variable(sample_inputs.reshape(n_class*(n_xs+n_xq), n_channles* n_size* n_size))
        #z = self.encoder.forward(sample_inputs) 
        z_dim = mu.size(-1)
        mu = mu.reshape(n_class, n_sam, z_dim)
        logvar = logvar.reshape(n_class, n_sam, z_dim)
        z_proto_mu = mu.mean(1)
        z_proto_logvar = logvar.mean(1)
        return z_proto_mu, z_proto_logvar
    
    def loss_initial(self, sample_inputs, oldimgs, oldprotos_mu, oldprotos_logvar, n_sam, n_curr_clas, n_clas, n_xs, n_xq, n_channles, n_size, temperature):
        
        sample_inputs = sample_inputs.reshape(n_curr_clas*n_sam, n_channles, n_size, n_size)
        #z: n_class, n_samples, n_channles*n_size*n_size
        z, mu, logvar = self.forward(sample_inputs, n_xs+n_xq)   
        z_dim = mu.size(-1)
        #logvar_mean = logvar.view(n_curr_clas, n_sam, z_dim).mean(1)
        
        n_support = n_xq + n_xs
        n_query = n_xq + n_xs
        
        split_nsam_support = int(math.floor(n_sam/2))
        split_nsam_query = n_sam - split_nsam_support
        
        target_inds = torch.arange(0, n_clas).view(n_clas, 1, 1).expand(n_clas, split_nsam_query*n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        #print('loss_initial -target inds')
        #print(target_inds.size())

        if z.is_cuda:
            target_inds = target_inds.cuda()
        
        z = z.reshape(n_curr_clas, n_sam, n_query, z_dim)

        z_proto = z[:,0:split_nsam_support,:,:].view(n_curr_clas, split_nsam_support*n_support, z_dim).mean(1)
        zq = z[:,split_nsam_support:,:,:].reshape(n_curr_clas * split_nsam_query * n_query, z_dim) #z[n_class*n_support:]
        #print('loss_initial - z_proto:')
        #print(z_proto.size())
        #print('loss_initial - zq:')
        #print(zq.size())
        
        if n_clas - n_curr_clas > 0:            
            ## for old imgs and old protos
            z_old_proto = self.reparameterize(oldprotos_mu, oldprotos_logvar, n_support).view(n_clas - n_curr_clas, n_support, z_dim).mean(1)
            oldimgs = oldimgs.reshape(split_nsam_query*(n_clas-n_curr_clas), n_channles, n_size, n_size)
            z_oldimgs, mu_oldimgs, logvar_oldimgs = self.forward(oldimgs, n_xs+n_xq)
            z_oldimgs = z_oldimgs.reshape( (n_clas - n_curr_clas)*split_nsam_query* n_query, z_dim)
            zq_combined = torch.cat((z_oldimgs, zq),0)
            #z_proto_logvar_combined = torch.cat((oldprotos_logvar, logvar_mean),0)
            z_proto_combined = torch.cat((z_old_proto, z_proto),0)
            
        else:
            zq_combined = zq
            z_proto_combined = z_proto
            #z_proto_logvar_combined = logvar_mean
        #print('loss_initial - zq_combined')
        #print(zq_combined.size())
        
        #print('loss_initial - z_proto_combined')
        #print(z_proto_combined.size())
        #print('loss_initial - z_proto_logvar_combined')
        #print(z_proto_logvar_combined.size())
        
        #z_proto_logvar_combined_std = torch.exp(-0.5*z_proto_logvar_combined)
        #z_proto_logvar_combined_std = F.softmax(z_proto_logvar_combined_std, 1)
        
        #print(z_proto_logvar_combined_std.sum(1))
        
        #z_proto: n_class, z_dim
        #zq: n_class*n_query, z_dim
        dists = euclidean_dist(zq_combined, z_proto_combined, temperature)

        #dists: n_class*n_query, n_class
        log_p_y = F.log_softmax(-dists, dim=1).view(n_clas, split_nsam_query*n_query, -1)

        #log_p_y: n_class, n_query, n_class (normalized from 0 to 1)
        #target_inds: n_class, n_query, 1
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        #print('loss_val')
        #print(loss_val.item())
        
        #KLDivergence loss from variational prototype
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #print('KLD')
        #print(KLD.item())
        #loss_val = loss_val + 0.0005*KLD
        #loss_val = KLD

        #pick the values of the ground truth index and calculate cross entropy loss
        _, y_hat = log_p_y.max(2)
        
        #y_hat: [n_class, n_query] ->  index of max value
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }
        
    def loss_val(self, sample_inputs, n_sam, n_clas, n_xs, n_xq, n_channles, n_size, temperature):
        
        sample_inputs = sample_inputs.reshape(n_clas*n_sam, n_channles, n_size, n_size)
        #z: n_class, n_samples, n_channles*n_size*n_size
        z, mu, logvar = self.forward(sample_inputs, n_xs+n_xq)   
        z_dim = mu.size(-1)
        
        n_support = n_xq + n_xs
        n_query = n_xq + n_xs
        
        split_nsam_support = int(math.floor(n_sam/2))
        split_nsam_query = n_sam - split_nsam_support
        
        target_inds = torch.arange(0, n_clas).view(n_clas, 1, 1).expand(n_clas, split_nsam_query*n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        #print('loss_val - target inds')
        #print(target_inds.size())
        #print(target_inds)

        if z.is_cuda:
            target_inds = target_inds.cuda()
        
        z = z.reshape(n_clas, n_sam, n_query, z_dim)

        z_proto = z[:,0:split_nsam_support,:,:].view(n_clas, split_nsam_support*n_support, z_dim).mean(1)
        zq = z[:,split_nsam_support:,:,:].reshape(n_clas * split_nsam_query * n_query, z_dim) #z[n_class*n_support:]
        #print('loss_val - z_proto:')
        #print(z_proto.size())
        #print('loss_val - zq:')
        #print(zq.size())
        #z_proto = z[:,0,:,:].view(n_clas, n_support, z_dim).mean(1)
        #zq = z[:,1,:,:].reshape(n_clas * n_query, z_dim) #z[n_class*n_support:]        
        
        zq_combined = zq
        z_proto_combined = z_proto
            
        logvar_mean = logvar.view(n_clas, n_sam, z_dim).mean(1)
        logvar_mean_std = torch.exp(-0.5*logvar_mean)
        logvar_mean_std = F.softmax(logvar_mean_std, 1)
        
        #z_proto: n_class, z_dim
        #zq: n_class*n_query, z_dim
        dists = euclidean_dist_weighted(zq_combined, z_proto_combined, logvar_mean_std, temperature)

        #dists: n_class*n_query, n_class
        log_p_y = F.log_softmax(-dists, dim=1).view(n_clas, split_nsam_query*n_query, -1)

        #log_p_y: n_class, n_query, n_class (normalized from 0 to 1)
        #target_inds: n_class, n_query, 1
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        #print('loss_val')
        #print(loss_val.item())

        #pick the values of the ground truth index and calculate cross entropy loss
        _, y_hat = log_p_y.max(2)
        
        #y_hat: [n_class, n_query] ->  index of max value
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }
        
    def loss_proto(self, sample_inputs, sampleproto_mu, sampleproto_logvar, n_xs, n_xq, n_class, n_channles, n_size, cuda, temperature):
        
        if cuda: sample_inputs = sample_inputs.cuda()
        if cuda: sampleproto_mu = sampleproto_mu.cuda()
        if cuda: sampleproto_logvar = sampleproto_logvar.cuda()
        
        zs = self.reparameterize(sampleproto_mu, sampleproto_logvar, n_xs+n_xq)
        
        sample_inputs = sample_inputs.reshape(n_class, n_channles, n_size, n_size)
        #z: n_class, n_samples, n_channles*n_size*n_size
        zq, mu, logvar = self.forward(sample_inputs, n_xs+n_xq)
        z_dim = mu.size(-1)
        n_support = n_xq + n_xs
        n_query = n_xq + n_xs
        
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        #print(target_inds)

        if zs.is_cuda:
            target_inds = target_inds.cuda()

        #print('zs')
        #print(zs.size())
        #print('zq')
        #print(zq.size())
        z_proto = zs.view(n_class, n_support, z_dim).mean(1)
        zq = zq.reshape(n_class * n_query, z_dim) #z[n_class*n_support:]    
        
        #sampleproto_logvar = sampleproto_logvar.view(n_class, z_dim)    
        #sampleproto_logvar_std = torch.exp(-0.5*sampleproto_logvar)
        #sampleproto_logvar_std = F.softmax(sampleproto_logvar_std, 1)
        
        #z_proto: n_class, z_dim
        #zq: n_class*n_query, z_dim
        #dists = euclidean_dist_weighted(zq, z_proto, sampleproto_logvar_std, temperature)
        dists = euclidean_dist(zq, z_proto, temperature)

        #dists: n_class*n_query, n_class
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        #log_p_y: n_class, n_query, n_class (normalized from 0 to 1)
        #target_inds: n_class, n_query, 1
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        #print('loss_val')
        #print(loss_val.item())
        
        #KLDivergence loss from variational prototype
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #print('KLD')
        #print(KLD.item())
        #loss_val = loss_val + 0.0005*KLD
        #loss_val = KLD

        #pick the values of the ground truth index and calculate cross entropy loss
        _, y_hat = log_p_y.max(2)
        
        #y_hat: [n_class, n_query] ->  index of max value
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }
        

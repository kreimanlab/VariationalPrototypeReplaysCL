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

class Protonet(nn.Module):
    def __init__(self, args):
        super(Protonet, self).__init__() 
        
        x_dim = args.x_dim
        hid_dim = args.hid_dim
        z_dim = args.z_dim
    
        self.fc1 = nn.Linear(x_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc31 = nn.Linear(hid_dim,  z_dim)
        self.fc32 = nn.Linear(hid_dim,  z_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)
    
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
        sample_inputs = sample_inputs.reshape(n_class*n_sam, n_channles* n_size* n_size)
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
          
    
    def loss_initial(self, sample_inputs, n_sam, n_xs, n_xq, n_class, n_channles, n_size, temperature):
        sample_inputs = sample_inputs.reshape(n_class*n_sam, n_channles* n_size* n_size)
        #z: n_class, n_samples, n_channles*n_size*n_size
        z, mu, logvar = self.forward(sample_inputs, n_xs+n_xq)   
        z_dim = mu.size(-1)
        
        n_support = n_xq + n_xs
        n_query = n_xq + n_xs
        
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        #print(target_inds)

        if z.is_cuda:
            target_inds = target_inds.cuda()
        
        z = z.reshape(n_class, n_sam, n_query, z_dim)

        z_proto = z[:,0,:,:].view(n_class, n_support, z_dim).mean(1)
        zq = z[:,1,:,:].reshape(n_class * n_query, z_dim) #z[n_class*n_support:]

        #z_proto: n_class, z_dim
        #zq: n_class*n_query, z_dim
        dists = euclidean_dist(zq, z_proto,temperature)

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
        
    def loss_proto(self, sample_inputs, sampleproto_mu, sampleproto_logvar, n_sam, n_xs, n_xq, n_class, n_channles, n_size, cuda, temperature):
        
        if cuda: sample_inputs = sample_inputs.cuda()
        if cuda: sampleproto_mu = sampleproto_mu.cuda()
        if cuda: sampleproto_logvar = sampleproto_logvar.cuda()
        
        zs = self.reparameterize(sampleproto_mu, sampleproto_logvar, n_xs+n_xq)
        
        sample_inputs = sample_inputs.reshape(n_class, n_channles* n_size* n_size)
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
            
        #z_proto: n_class, z_dim
        #zq: n_class*n_query, z_dim
        dists = euclidean_dist(zq, z_proto,temperature)

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
        
        
    
##########################  END OF MODEL DEFINITION  ##########################


import numpy as np
import torch
import torch.nn.functional as F
from base import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F

class Classifier(nn.Module):
    def __init__(self, vae, z_dim, clses, requires_grad=True, label=1, name ="Classifier", y_dim=0,input_channel_size=1,device ='cpu'):
        super().__init__()
        self.vae = vae
        if requires_grad == False:
            for p in self.parameters():
                p.requires_grad = False
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.device =device
        self.name =name
        self.cls = clses
        self.Label = label

        self.lambda_cos = 0.1
        self.lambda_ip = 2
        self.Linear1 = nn.Linear(z_dim,64)
        self.dropout = nn.Dropout(p=0.3)

        if self.cls== 2:
            self.Linear2 = nn.Linear(64,1).to(self.device)
        else:
            self.Linear2 = nn.Linear(z_dim,self.cls).to(self.device)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.Sigmoid = nn.Sigmoid()
        torch.nn.init.kaiming_normal_(self.Linear1.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.Linear2.weight, a=0, mode='fan_in', nonlinearity='relu')

    def classify(self, x, y):
        z = self.vae.enc.encode_nkl(x)

        h = self.Linear1(z)
        h = F.relu(h)

        h = self.Linear2(h)
        h = h.squeeze(dim=-1)        

        pred = self.Sigmoid(h).detach()
        pred = [self.Label if pred[i] >=0.5 else 0 for i in range(len(pred))]

        res = [1 if pred[i]==y[i] else 0 for i in range(len(pred))]
        acc = sum(res)/len(pred)

        return acc, len(pred), pred, res           
    def forward(self, x, y):
        z = self.vae.enc.encode_nkl(x)
        # h = F.relu(z)
        # h = self.dropout(h)
        h = self.Linear1(z)
        h = F.relu(h)
        
        h = self.Linear2(h)
        h = h.squeeze(dim=-1)
        loss = self.bce(input=h, target=y)
        pred = self.Sigmoid(h).detach()
        pred = [self.Label if pred[i] >=0.5 else 0 for i in range(len(pred))]
        print('total:',len(pred),'acc for batch:', sum([1 if pred[i]==y[i] else 0 for i in range(len(pred))])/len(pred))
        return loss.mean()


class Distance_Relu_Loss(nn.Module):
    def __init__(self, z_dim, requires_grad=True, name ="relu_loss", y_dim=0,input_channel_size=1,device ='cpu'):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.device =device
        self.name =name


        self.lambda_cos = 0.1
        self.lambda_ip = 2

        self.weights = torch.nn.Parameter(torch.randn(1,1,z_dim,dtype=torch.float, requires_grad=requires_grad).to(device))
        self.weights.data = self.weights.data.detach() / (self.weights.data.detach().norm() + 1e-10)
        nn.Module.register_parameter(self,'d_weights',self.weights)

        
    

    def forward(self, z1,z2):
        

        zn12 =(z1 -z2).norm(dim=1)
        z_norm = zn12
        print(z_norm.detach())
        z1 = torch.unsqueeze(z1, dim=1)
        z2 = torch.unsqueeze(z2, dim=1)

        self.weights.data = self.weights.data.detach()/(self.weights.data.detach().norm() + 1e-10)

        h1 = F.conv1d(z1, self.weights)
        h2 = F.conv1d(z2, self.weights)
        h1 = torch.squeeze(h1, dim=1)
        h2 = torch.squeeze(h2, dim=1)
        ncos12 = (h1-h2).squeeze() / (zn12 + 1e-7)
        return ncos12


class Encoder(nn.Module):
    def __init__(self, z_dim, y_dim=0,input_channel_size=1,device ='cpu'):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.device =device
        self.net = nn.Sequential(
            nn.Conv3d(input_channel_size,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.Linear1 = nn.Linear(1024,512).to(self.device)
        self.Linear2 = nn.Linear(1024,1024).to(self.device)
        self.Linear3 = nn.Linear(1024,2*z_dim).to(self.device)
        self.net =self.net.float()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.Linear4 = nn.Linear(1024,z_dim).to(self.device)


    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        # print(xy.dtype)
        h = self.net(xy.float())
        batch_size = h.shape[0]
        # print(h.shape)
        h = h.reshape(batch_size,-1)

        h = self.Linear3(h)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

    def encode_nkl(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        # print(xy.dtype)
        h = self.net(xy.float())
        batch_size = h.shape[0]
        # print(h.shape)
        h = h.reshape(batch_size,-1)

        h = self.Linear4(h)
        h = F.tanh(h)
        return h





class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0, fp_dim=1024,output_channel_size=1,device ='cpu'):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.device =device
        self.Linear1 = nn.Linear(z_dim + y_dim, 1024).to(self.device)
        self.Linear2 = nn.Linear(1024,1024).to(self.device)
        self.Linear3 = nn.Linear(z_dim, fp_dim).to(self.device)
        self.dropout1 = nn.Dropout(p=0.5)
        # self.Linear = nn.Linear(z_dim,fp_dim)

        self.net = nn.Sequential(

            nn.ConvTranspose3d(16,16,3,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose3d(16,64,3,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose3d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.ConvTranspose3d(16, output_channel_size, 3, padding=1),

        )
        self.net = self.net.float()

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)

        h = self.Linear3(zy)
        h = F.tanh(h)
        h = h.reshape(h.shape[0],16,4,4,4)
        return self.net(h)

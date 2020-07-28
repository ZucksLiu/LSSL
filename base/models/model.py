import torch
from base import utils as ut
from base.models import nns
from torch import nn
from torch.nn import functional as F
import numpy as np


class MultipleTimestepLSTM(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(64,64,64), z_dim=512,inter_num_ch=16, fc_num_ch=16, lstm_num_ch=16, kernel_size=3, name ='LSTM',
                conv_act='relu', requires_grad=True,fc_act='tanh', num_cls=2, num_timestep=5, skip_missing=True, init_lstm=False, rnn_type='GRU', fe_arch='AE', vae =None):
        super(MultipleTimestepLSTM, self).__init__()
        self.name =name
        self.z_dim=z_dim
        self.fe_arch = fe_arch
        if fe_arch == 'AE':
            self.feature_extractor = vae
            num_feat = 512
        elif fe_arch == 'VAE':
            self.feature_extractor = vae
            num_feat = 512
        if fc_act == 'tanh':
            fc_act_layer = nn.Tanh()
        elif fc_act == 'relu':
            fc_act_layer = nn.ReLU()
        else:
            raise ValueError('No implementation of ', fc_act)
        if requires_grad== False:
            for p in self.parameters():
                p.requires_grad = False
        if num_cls == 2 or num_cls == 0:
            num_output = 1
        else:
            num_output = num_cls
        self.num_cls = num_cls
        self.dropout_rate = 0.1
        self.skip_missing = skip_missing
        self.fc1 = nn.Sequential(
                        nn.Linear(num_feat, fc_num_ch),
                        fc_act_layer)

        # self.fc1 = nn.Sequential(
        #                 nn.Linear(num_feat, 4*fc_num_ch),
        #                 fc_act_layer,
        #                 nn.Dropout(self.dropout_rate))

        # self.fc2 = nn.Sequential(
        #                 nn.Linear(4*fc_num_ch, fc_num_ch),
        #                 fc_act_layer),
        #                 nn.Dropout(self.dropout_rate))
        # nn.Sequential(
        #     nn.Linear(512,16),
        #     nn.GRU(input_ize=16,hidden_size=16,
        #     num_layers=1,batch_first=True)
        #     nn.Linear(16,1)
        # )

        if rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        elif rnn_type == 'GRU':
            self.lstm = nn.GRU(input_size=fc_num_ch, hidden_size=lstm_num_ch, num_layers=1,
                            batch_first=True)
        else:
            raise ValueError('No RNN Layer!')

        self.fc3 = nn.Linear(lstm_num_ch, num_output)

        if init_lstm:
            self.init_lstm()

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1.0)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, mask):
        #pdb.set_trace()
        bs, ts = x.shape[0], x.shape[1]
        x = torch.cat([x[b,...] for b in range(bs)], dim=0)  # (bs,ts,32,64,64) -> (bs*ts,32,64,64)
        x = x.unsqueeze(1)  # (bs*ts,1,32,64,64)
        if self.fe_arch == 'AE':

            out_z = self.feature_extractor.enc.encode_nkl(x)   # (bs*ts,512)
        elif self.fe_arch == 'VAE':
            z_param_m, z_param_v = self.feature_extractor.enc.encode(x)
            # print('sampling')
            # out_z = ut.sample_gaussian(z_param_m,z_param_v)
            out_z = z_param_m            

        fc1 = self.fc1(out_z)
        # fc2 = self.fc2(fc1) # (bs*ts,16)
        fc2_concat = fc1.view(bs, ts, -1)  # (bs, ts, 16)

        if self.skip_missing:
            num_ts_list = mask.sum(1)
            if (num_ts_list == 0).sum() > 0:
                pdb.set_trace()

            _, idx_sort = torch.sort(num_ts_list, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            num_ts_list_sorted = num_ts_list.index_select(0, idx_sort)
            fc2_concat_sorted = fc2_concat.index_select(0, idx_sort)
            fc2_packed = torch.nn.utils.rnn.pack_padded_sequence(fc2_concat_sorted, num_ts_list_sorted, batch_first=True)
            lstm_packed, _ = self.lstm(fc2_packed)
            lstm_sorted, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_packed, batch_first=True)
            lstm = lstm_sorted.index_select(0, idx_unsort)
            # print(lstm.shape)
        else:
            lstm, _ = self.lstm(fc2_concat) # lstm: (bs, ts, 16)
        if lstm.shape[1] != ts:
            pad = torch.zeros(bs, ts-lstm.shape[1], lstm.shape[-1])
            lstm = torch.cat([lstm, pad.cuda()], dim=1)

        output = self.fc3(lstm)
        if self.num_cls == 0:
            output = F.relu(output)
        if self.skip_missing:
            tpm = [output[i, num_ts_list[i].long()-1, :].unsqueeze(0) for i in range(bs)]
            output_last = torch.cat(tpm, dim=0)
            return [output_last, output, fc2_concat]
        else:
            return [output[:,-1,:], output]


class Siamese_Network_v2(nn.Module):
    def __init__(self, vae, relu_loss, name="SiameseNet_weight_AE", lambda_relu=2, z_dim=16,device ='cpu'):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.device = device

        self.name =name
        self.vae = vae
        self.relu_loss =relu_loss

        self.lambda_cos = 1


    def loss_nkl(self, x):
        x1 = torch.unsqueeze(x[:,0,:,:,:],axis=1)
        x2 = torch.unsqueeze(x[:,1,:,:,:],axis=1)
        
        vae_loss_x1, summaries_1, z1 = self.vae.loss_nkl(x1)
        vae_loss_x2, summaries_2, z2 = self.vae.loss_nkl(x2)
        

        ncos12 = self.relu_loss(z1,z2)
        loss_cos = 1 + (ncos12)
        print('loss_cos:',loss_cos)
        vae_loss = vae_loss_x1 + vae_loss_x2
        reg_loss = self.lambda_cos * loss_cos.mean()
        print('vae_loss:', vae_loss)
        print('reg_loss:', reg_loss)

        loss = vae_loss + reg_loss 
        return loss, summaries_1, reg_loss.detach(), vae_loss.detach()

''' Please look at loss_nkl and negatove_elbo_bound_ae'''
class VAE3d(nn.Module):
    def __init__(self, nn='v8', name='vae3d', z_dim=16,device ='cpu', lambda_kl =0.01):
        super().__init__()
        self.name = name
        self.z_dim = z_dim

        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, device =device)
        self.dec = nn.Decoder(self.z_dim, device =device)
        self.lambda_kl = lambda_kl
        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)


    def negative_elbo_bound_cos(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        z_param_m, z_param_v = self.enc.encode(x)
        z = ut.sample_gaussian(z_param_m,z_param_v)
        xhat = self.dec.decode(z)

        rec = ut.mseloss(x,xhat)

        kl = ut.kl_normal(z_param_m,z_param_v,self.z_prior_m,self.z_prior_v)
        # print(kl.shape)

        nelbo = rec + self.lambda_kl * kl

        nelbo = nelbo.mean()
        kl = kl.mean() 
        rec = rec.mean()
        print(nelbo,kl,rec)

        return nelbo, kl, rec, z_param_m, z_param_v
    
    def negative_elbo_bound_ae(self, x):
        """
        Using in loss_nkl

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
            z: tensor: (): encoded representation
        """
        z = self.enc.encode_nkl(x)
        xhat = self.dec.decode(z)
        rec = ut.mseloss(x,xhat)

        nelbo = rec 

        nelbo = nelbo.mean()

        kl = 0
        z_param_m = z_param_v = 0
        print(rec)
        rec = rec.mean()
        print(nelbo,rec)
        return nelbo, kl, rec, z




    def loss(self, x):
        nelbo, kl, rec, z_param_m, z_param_v = self.negative_elbo_bound_cos(x)
        # nelbo, kl, rec, z_param_m, z_param_v = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries, z_param_m, z_param_v
    
    def loss_nkl(self, x):
        nelbo, kl, rec, z = self.negative_elbo_bound_ae(x)
        # nelbo, kl, rec, z_param_m, z_param_v = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries, z   

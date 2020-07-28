
import argparse
import numpy as np
import torch
import tqdm
import os
import pickle as pkl
import scipy as sp
from base import utils as ut

from base.train import train3d_Siamese
from base.train import train3d_cls, train3d_cls_rnn
from base.models.nns.v13 import Classifier
from pprint import pprint
from torchvision import datasets, transforms

import joblib
from base.models.model import MultipleTimestepLSTM



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=512,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=1560, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=50, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=37,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=16,     help="Flag for training")
parser.add_argument('--Type',     type=str, default='Run_cls_new',     help="Flag for training")
''' Type: Run_all_label_time_tri/Run_all_time/ Run_only_normal_time, Run_4_time, Run_all_label_time, Run_cls'''
parser.add_argument('--iter_restart', type=int, default=4800, help="Save model every n iterations")
parser.add_argument('--BATCH_SIZE',     type=int, default=64,     help="Flag for training")
parser.add_argument('--iter_load',     type=int, default=1,     help="Flag for loading version of model")
parser.add_argument('--Siamese',     type=str, default='SiameseNetTri', help="SiameseNetTri\SiameseNetAE\SiameseNet\SiameseNetW\SiamgeseNetAEReg")

cls_list= [0,2] # [0,1] -- [NC.MCI]

global_step = 1500
print('use_s_region:',use_s_region)

''' Initialize model layout'''
args = parser.parse_args()

if args.Siamese == 'SiameseNetTri':
    from base.models.model import VAE3d, Siamese_Network_v2
    from base.models.nns.v13 import Distance_Relu_Loss


Type = args.Type
vae3d_layout = [
    ('model={:s}',  'vae3d'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
vae3d_model_name = '_'.join([t.format(v) for (t, v) in vae3d_layout])

relu_loss_layout = [
    ('model={:s}',  'relu_loss'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
relu_loss_model_name = '_'.join([t.format(v) for (t, v) in relu_loss_layout])

SiameseNet_layout = [
    ('model={:s}',  args.Siamese),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
SiameseNet_model_name = '_'.join([t.format(v) for (t, v) in SiameseNet_layout])

Classifier_layout = [
    ('model={:s}',  'Classfier'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
Classifier_model_name = '_'.join([t.format(v) for (t, v) in Classifier_layout])


pprint(vars(args))
print('Model name:', SiameseNet_model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' Set random seed'''
ut.set_seed(2020)

path ="/scratch/users/zucks626/ADNI/ADNI_Longitudinal_1styear/raw_image/"
path_saveimg = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_1styear/save_image/"

path_1 = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_2ndyear/raw_image/"
path_2 = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_1styear/raw_image/"
path2_saveimg = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_2ndyear/save_image/"
path_all = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_all/raw_image/"
pathall_saveimg = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_all/save_image/"

pathallnew_img='/scratch/users/zucks626/ADNI/ADNI_Longitudinal_all_new/img_64_longitudinal/raw_image/'

if Type == 'Run_cls_new':
    ''' Extract data from pickle file'''
    f = open(pathall_saveimg + "augment_pair_cls_AD.pkl","rb")
    pair = pkl.load(f)
    f.close()
    f = open(pathall_saveimg + "augment_d_cls_AD.pkl","rb")
    dataset = pkl.load(f)
    f.close()    
    f = open(pathall_saveimg + "augment_label_cls_AD.pkl","rb")
    label = pkl.load(f)
    f.close()  
    id_idx, cal_idx = ut.get_idx_label(label)
    pair_new, label_new = ut.get_pair_idx_label_new(id_idx,pair, cls_list)
    print(pair_new)
    print(label_new)
    print(len(pair_new))
    

                      
elif Type == 'Run_all_label_time_tri_new':
    ''' Extract data from pickle file'''

    ###################### store Normalized Dataset from dataset  #####################
    f = open(pathall_saveimg + "pair_all_clean.pkl","rb")
    pair = pkl.load(f)
    f.close()     
    print(pair)
    f = open(pathall_saveimg + "dataset_all_clean_before_D.pkl","rb")
    dataset_intmd = pkl.load(f)
    f.close()
    f = open(pathall_saveimg + "Id_attr.pkl","rb")
    Id_attr = pkl.load(f)
    f.close()
    print(dataset_intmd.shape, dataset_intmd.dtype)
    new_d,new_l,new_pair = ut.augment_by_subject(dataset_intmd,pair,Id_attr,1200)

    new_d_AD, new_l_AD,new_pair_AD = ut.augment_by_subject_label(dataset_intmd,pair,Id_attr,1000)
    orig_d, orig_l,orig_pair = ut.augment_by_subject(dataset_intmd,pair,Id_attr,100)

    final_d = torch.cat([orig_d,new_d,new_d_AD])
    final_l = orig_l + new_l + new_l_AD
    final_pair = ut.append_pair(orig_pair,new_pair)
    final_pair = ut.append_pair(final_pair,new_pair_AD)
    print(final_pair)
    print(len(final_l))

    f = open(pathall_saveimg + "augment_pair_cls_AD.pkl","wb")
    pkl.dump(final_pair, f)
    f.close()
    f = open(pathall_saveimg + "augment_d_cls_AD.pkl","wb")
    pkl.dump(final_d, f,protocol=4)
    f.close()    
    f = open(pathall_saveimg + "augment_label_cls_AD.pkl","wb")
    pkl.dump(final_l, f)
    f.close()  


if args.Type == 'Run_cls_new':

    train_loader_list, train_label_loader_list = ut.split_dataset_folds_new_true_subject(dataset, label_new, pair_new,folds=5, BATCH_SIZE = args.BATCH_SIZE, shuffle=True,seed=2020)
        
    
        

vae = VAE3d(z_dim=args.z, name=vae3d_model_name, device=device, nn='v13')


if args.train == 14: 
    writer = ut.prepare_writer(vae3d_model_name, overwrite_existing=True)
    ''' /scratch/users/zucks626/ADNI/ae_relu/checkpoints/'''
    ''' Load pre-trained vanilla VAE model'''
    ut.load_model_by_name_ae(vae, global_step=global_step, device=device)
    # print(vae.state_dict())
    # SN =Siamese_Network(vae,relu_loss,name = SiameseNet_model_name,z_dim=args.z,device =device,lambda_relu=1).to(device)
    Cls =Classifier(vae, clses =2, name = Classifier_model_name, label=1, z_dim=args.z,device =device,requires_grad=True).to(device)
    print(Cls.parameters())
    train3d_cls(model=Cls, dataset = train_loader_list, label=train_label_loader_list,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          lr_decay_step=360, schedule=False,
          iter_max=args.iter_max,iter_restart=global_step,
          iter_save=args.iter_save,
          requires_grad=True,vae=None)


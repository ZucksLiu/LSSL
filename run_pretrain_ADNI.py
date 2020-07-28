
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

from pprint import pprint
from torchvision import datasets, transforms

import joblib
from base.models.model import MultipleTimestepLSTM



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=512,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=1000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=50, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=73,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=15,     help="Flag for training")
parser.add_argument('--Type',     type=str, default='Run_all_label_time_tri_new',     help="Flag for training")
''' Type: Run_all_label_time_tri/Run_all_time/ Run_only_normal_time, Run_4_time, Run_all_label_time, Run_cls'''
parser.add_argument('--iter_restart', type=int, default=4800, help="Save model every n iterations")
parser.add_argument('--BATCH_SIZE',     type=int, default=64,     help="Flag for training")
parser.add_argument('--iter_load',     type=int, default=1,     help="Flag for loading version of model")
parser.add_argument('--Siamese',     type=str, default='SiameseNetTri', help="SiameseNetTri\SiameseNetAE\SiameseNet\SiameseNetW\SiamgeseNetAEReg")

cls_list= [0,2] 

global_step = 1600

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

RNN_layout = [
    ('model={:s}',  'RNN'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
RNN_model_name = '_'.join([t.format(v) for (t, v) in RNN_layout])




pprint(vars(args))
print('Model name:', SiameseNet_model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
if device != 'cpu':
    print(1)

path ="/scratch/users/zucks626/ADNI/ADNI_Longitudinal_1styear/raw_image/"
path_saveimg = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_1styear/save_image/"

path_1 = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_2ndyear/raw_image/"
path_2 = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_1styear/raw_image/"
path2_saveimg = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_2ndyear/save_image/"
path_all = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_all/raw_image/"
pathall_saveimg = "/scratch/users/zucks626/ADNI/ADNI_Longitudinal_all/save_image/"

pathallnew_img='/scratch/users/zucks626/ADNI/ADNI_Longitudinal_all_new/img_64_longitudinal/raw_image/'

pathallnew_adni = '/scratch/users/zucks626/ADNI/ADNI_Longitudinal_all_new/img_64_longitudinal/'
pathallnew_labdata='/scratch/users/zucks626/ADNI/lab_data/img_64_longitudinal/'


if Type == 'Run_all_label_time_tri_new':
    ''' Extract data from pickle file'''
    '''  store lab data and pair adni data '''
    ###################   May 28 process and store lab data ###### 
    data_type = 'adni'
    # if data_type == 'lab':
    #     path_idxfile='/home/users/zucks626/miccai/lab_data/'
    #     data_path = pathallnew_labdata
    #     dataset_1 = ut.get_dataset_from_idx_file(path_idxfile+'img1.txt', data_path, data=data_type)
    #     dataset_2 = ut.get_dataset_from_idx_file(path_idxfile+'img2.txt', data_path, data=data_type)
    #     print(dataset_1.shape)
    #     dataset_1 = torch.tensor(dataset_1)
    #     dataset_2 = torch.tensor(dataset_2)
    #     dataset = torch.cat([dataset_1,dataset_2],axis=1)
    #     print(dataset.shape)
    #     f = open(pathall_saveimg + "dataset_pair_realone_labdata.pkl","wb")
    #     pkl.dump(dataset, f,protocol=4)
    #     f.close()
    # elif data_type == 'adni':
    #     path_idxfile='/home/users/zucks626/miccai/code_latest/'
    #     data_path = pathallnew_adni
    #     path_remove_file = path_idxfile
    #     remove_file_list = np.genfromtxt(path_remove_file+'failed_processing.txt', dtype='str') 
    #     dataset = ut.get_dataset_from_idx_file_adni(path_idxfile+'img1.txt', path_idxfile+'img2.txt', data_path, remove_file_list=remove_file_list, data=data_type)
    #     dataset = torch.tensor(dataset)
    #     print(dataset.shape)
    #     f = open(pathall_saveimg + "dataset_pair_realone_adni.pkl","wb")
    #     pkl.dump(dataset, f,protocol=4)
    #     f.close()
    # sleep(1000)
    # #############  load temp Dataset and temp lookup_label and mean_std##########
    # f = open(pathall_saveimg + "Dataset_all_clean_temp.pkl","rb")
    # Dataset = pkl.load(f)
    # f.close()
    # f = open(pathall_saveimg + "lookup_label_all_clean_temp.pkl","rb")
    # lookup_label = pkl.load(f)
    # f.close()
    # f = open(pathall_saveimg + "mean_std_clean.pkl","rb")
    # mean_std = pkl.load(f)
    # f.close()
    # f = open(pathall_saveimg + "Id_attr.pkl","rb")
    # Id_attr = pkl.load(f)
    # f.close()
    # mean_given, std_given = mean_std
    # print(mean_given, std_given)
    # ######################################################

    if args.train >0 or use_s_region == True:
        if data_type =='lab':
            f = open(pathall_saveimg + "dataset_pair_realone_labdata.pkl","rb")
        elif data_type =='adni':
            f = open(pathall_saveimg + "dataset_pair_realone_adni.pkl","rb")
        dataset = pkl.load(f)
        f.close()
        print("load dataset")
        print("Shape of datadset:", dataset.size())
        dataset_mean = 0
        dataset_std = 1



train_loader = ut.split_dataset_realone(dataset, BATCH_SIZE = args.BATCH_SIZE)


vae = VAE3d(z_dim=args.z, name=vae3d_model_name, device=device, nn='v13')
relu_loss = Distance_Relu_Loss(z_dim=args.z,name=relu_loss_model_name, device=device,requires_grad=True)

writer = ut.prepare_writer(vae3d_model_name, overwrite_existing=True)
''' /scratch/users/zucks626/ADNI/ae_relu/checkpoints/'''
''' Load pre-trained vanilla VAE model'''
SN =Siamese_Network_v2(vae,relu_loss,name = SiameseNet_model_name,z_dim=args.z,device =device,lambda_relu=1).to(device)
ut.load_model_by_name_ae(SN, global_step=global_step, device=device)
train3d_Siamese(model=SN,
          train_loader=train_loader,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          lr_decay_step=400,
          iter_max=args.iter_max,
          iter_save=args.iter_save)

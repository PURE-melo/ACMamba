import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.io as scio
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat, savemat
from scipy.ndimage import zoom
from skimage.segmentation import slic
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.distributions import Multinomial
from torch.nn.functional import gumbel_softmax, interpolate
from torch.utils.data import DataLoader, TensorDataset

from utils import PCGrad
from spectral import *
from rsal import RSAL_network
import torch.nn.init as init
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class ACMamba(nn.Module):
        def __init__(self):
            super(ACMamba, self).__init__()
            self.rsal_model = RSAL_network(
                in_chans=band,
                patch_size=1,
                patchH=H,
                patchW=W,
                depths=[1],
                dims=[hidden_dim],
                drop_path_rate=0.1,
                attn_drop_rate=0.0
            )
            self.rsal_model_region = RSAL_network(
                in_chans=band,
                patch_size=1,
                patchH=H,
                patchW=W,
                depths=[1],
                dims=[hidden_dim],
                drop_path_rate=0.1,
                attn_drop_rate=0.0
            )
            self.decoder = RSAL_network(
                in_chans=hidden_dim,
                patch_size=1,
                patchH=H,
                patchW=W,
                depths=[1],
                dims=[band],
                drop_path_rate=0.1,
                attn_drop_rate=0.0
            )

        def forward(self, x, mask=None):
    
            if mask !=None:
                encoder = self.rsal_model(x).permute(0,3,1,2)
                decoder = self.decoder(encoder).permute(0,3,1,2)
                region_encoder = self.rsal_model_region(x*mask).permute(0,3,1,2)
                region_decoder = self.decoder(region_encoder).permute(0,3,1,2)
                return decoder, region_decoder
            else:
                encoder1 = self.rsal_model(x).permute(0,3,1,2)
                decoder1 = self.decoder(encoder1).permute(0,3,1,2)
                encoder2 = self.rsal_model_region(x).permute(0,3,1,2)
                decoder2 = self.decoder(encoder2).permute(0,3,1,2)
            return (decoder1+decoder2)/2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mean_auc = []
mean_time = []
setup_seed(4)
for fpath in ['./data/abu-urban-2.mat', './data/abu-urban-5.mat', './data/AVIRIS-3.mat', './data/SanDiego.mat','./data/Hydice.mat','./data/Hyperion.mat', './data/Cri.mat']:
    mat_dt = loadmat(fpath)
    anomaly_actual = mat_dt['map']
    gt = (anomaly_actual > 0).astype(int)
    gt_flat = gt.flatten()
    start_time = time.time()
    data_array = mat_dt['data'].astype(np.float32)
    ori_H, ori_W, band = data_array.shape
    scaler = preprocessing.MinMaxScaler()
    data_normalized = scaler.fit_transform(data_array.reshape(data_array.shape[0] * data_array.shape[1], -1))
    data_normalized = data_normalized.reshape(data_array.shape)
    data_normalized = torch.tensor(data_normalized, dtype=torch.float)  
    data_normalized = np.transpose(data_normalized, axes=(2, 0, 1))
    data_normalized = data_normalized.unsqueeze(0) 
    _, _, H, W =  data_normalized.shape

    n_segments = int(ori_H*ori_W / 150)
    
    segments = slic(data_normalized.squeeze(0).permute(1, 2, 0).cpu().numpy(), n_segments=n_segments, compactness=1, sigma=0.5)
    
    segments = torch.from_numpy(segments).unsqueeze(0).unsqueeze(0).long()
    segments = segments
    n_segments = segments.max()

    data_bank = torch.zeros((data_normalized.shape[1], n_segments))
    max_bank = torch.zeros((data_normalized.shape[1], n_segments))
    min_bank = torch.zeros((data_normalized.shape[1], n_segments))
    std_bank = torch.zeros((data_normalized.shape[1], n_segments))
    for i in range(n_segments):
        data_bank[:, i] = data_normalized[0, :, segments[0, 0] == (i+1)].mean(1)
        max_bank[:, i], _ = data_normalized[0, :, segments[0, 0] == (i+1)].max(1)
        min_bank[:, i], _ = data_normalized[0, :, segments[0, 0] == (i+1)].min(1)
        std_bank[:, i] = data_normalized[0, :, segments[0, 0] == (i+1)].std(1)
    data_bank = data_bank.unsqueeze(0).unsqueeze(-1).cuda()
    max_bank = max_bank.unsqueeze(0).unsqueeze(-1).cuda()
    min_bank = min_bank.unsqueeze(0).unsqueeze(-1).cuda()
    std_bank = std_bank.unsqueeze(0).unsqueeze(-1).cuda()
    H = n_segments
    W = 1

    hidden_dim = 256
    

    start_train_time = time.time()
    model = ACMamba().to(device)

    criterion_ae = nn.MSELoss()
    optimizer_ae = PCGrad(torch.optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.99), weight_decay=5e-4))

    data_loader = DataLoader(data_bank, batch_size=1, shuffle=True)
    loss_values = []
    region_bank = torch.zeros((n_segments)).to(device)
    masked_regions = []
    mse = None 

    for epoch in range(100):
        for i, inputs in enumerate(data_loader, 0):
            inputs = inputs.to(device)
            inputs = inputs + 2*(2*torch.rand(inputs.shape[2]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(-1)-1)*std_bank
            inputs = inputs.clamp(min_bank, max_bank)

            optimizer_ae.zero_grad()
            ae_list = []
            mask = torch.ones_like(data_bank).to(device)
            for region in masked_regions:
                mask[0, :, region-1] = 0
            if mse != None:
                reconstructed, reconstructed_mask = model(inputs, mask)
            else:
                reconstructed, reconstructed_mask = model(inputs, mask)
            ae_list.append(criterion_ae(reconstructed_mask, inputs*mask))
            ae_list.append(criterion_ae(reconstructed, inputs*mask))
            with torch.no_grad():
                mse = ((reconstructed - inputs) ** 2).mean(dim=1)
                for i in range(n_segments):
                    region_bank[i] += mse[0, i].mean()
                masked_regions = random.choices([x for x in range(1, n_segments+1)], weights=[r.detach().item() for r in region_bank], k=int(n_segments*0.01))
            optimizer_ae.pc_backward(ae_list)
            optimizer_ae.step() 

    end_train_time = time.time()
    train_time_cost = end_train_time - start_train_time

    data_tensor = data_bank.cuda()
    base = data_normalized.cuda()
    start_test_time = time.time()
    with torch.no_grad():
        decoded_data = model(data_normalized.to(device).reshape(1,-1,ori_H*ori_W,1))
        reconstruction_error1 = ((base - decoded_data.reshape(1,-1,ori_H,ori_W)) ** 2).mean(dim=1)
        decoded_data2 = model(data_tensor)
        real_decoded_data = torch.zeros_like(base).to(device)
        for i in range(n_segments):
            real_decoded_data[0, :, segments[0, 0] == (i+1)] = decoded_data2[0, :, i]
        reconstruction_error2 = torch.tensor(rx((((base - real_decoded_data) ** 2)).squeeze().permute(1,2,0).cpu().numpy())).unsqueeze(0).cuda()
        reconstruction_error = (reconstruction_error1)*(reconstruction_error2)
    reconstruction_error_squeezed = reconstruction_error.squeeze(0)
    anomaly_actual = mat_dt['map']
    gt = (anomaly_actual > 0).astype(int)
    gt_flat = gt.flatten()
    reconstruction_error_np = reconstruction_error_squeezed.cpu().numpy()
    reconstruction_error_np = reconstruction_error_np.flatten()

    threshold = 0.0
    predictions = (reconstruction_error_np > threshold).astype(int)
    predictions_flat = predictions.flatten()

    accuracy = accuracy_score(gt_flat, predictions_flat)

    f1 = f1_score(gt_flat, predictions_flat)
    auc = roc_auc_score(gt_flat, reconstruction_error_np)
    mean_auc.append(auc)
    end_test_time = time.time()
    test_time_cost = end_test_time - start_test_time
    cost = train_time_cost+test_time_cost
    print(f"Dataset: {os.path.basename(fpath)}")
    print(f"AUC: {auc:.4f}")
    print(f"Time Cost: {cost:.2f}s")
    print("-" * 60)
    
    reconstruction_error_np = reconstruction_error_np - reconstruction_error_np.min()
    reconstruction_error_np = reconstruction_error_np / reconstruction_error_np.max()
    mean_time.append(cost)
    torch.cuda.empty_cache()

    
print("\n" + "=" * 85)
print("FINAL RESULTS:")
print("=" * 85)
print(f"All AUC scores: {[f'{score:.4f}' for score in mean_auc]}")
print(f"All Time costs: {[f'{time:.2f}s' for time in mean_time]}")
print(f"Overall Mean AUC: {np.mean(mean_auc):.4f}")
print("=" * 85)


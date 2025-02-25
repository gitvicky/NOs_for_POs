from matplotlib import pyplot as plt 
import numpy as np 
import torch 
from torch.utils.data import Dataset
import h5py 
import glob 
from tqdm import tqdm
import os 
import shutil 

tmp_loc = os.getcwd()
# try: 
#     shutil.rmtree(tmp_loc)
#     os.mkdir(tmp_loc)
# except:
#     pass


from data_loaders import * 

n_sims = 100
data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Code/Neural_PDE/Data'
data =  np.load(data_loc + '/NS_FV_combined.npz')
u = data['u'].astype(np.float32)[:n_sims]
v = data['v'].astype(np.float32)[:n_sims]
p = data['p'].astype(np.float32)[:n_sims]
rho = data['rho'].astype(np.float32)[:n_sims]
dx = data['dx']
x = np.linspace(0, 1, 128)

dt = data['dt']
dt = torch.tensor(dt, dtype=torch.float)

fields = stacked_fields([u,v,p,rho])

for ii in range(4):
    plt.figure()
    plt.hist(fields[:,ii].flatten(), bins=100)
    plt.savefig(tmp_loc + '/field_' + str(ii) + '.png')

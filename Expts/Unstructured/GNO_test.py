#Imports

# %% 

import shutil
import os
import yaml 
import argparse
from timeit import default_timer
import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn 
#Setting up the seeds and devices
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

model_type = 'nnconv'
# %% 
#Loading config
def is_notebook():
    """Check if running in Jupyter notebook"""
    try:
        get_ipython()
        return True
    except NameError:
        return False

def load_config(config_path='configs/train/Wave_GNO.yaml'):
    """Load configuration from YAML file"""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        configuration = yaml.safe_load(f)
    print(f"Configuration loaded successfully!")
    return configuration

# Determine config path based on environment
if is_notebook():
    # Jupyter notebook mode - set your config path here
    CONFIG_PATH = '/pitagora/home/userexternal/vgopakum/NOs_for_POs/Expts/configs/train/Wave_GNO.yaml'  # Change this as needed
    configuration = load_config(CONFIG_PATH)
else:
    # Command-line mode
    import argparse
    parser = argparse.ArgumentParser(description='Training script with YAML config')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    args = parser.parse_args()
    configuration = load_config(args.config)

# %% 
#Loading dataset
from data_loaders import *
pde = configuration['Physics']['pde']
if pde == 'Incomp. Navier-Stokes':
    fields, X, Y, dt, mass, params, edge_attr, edge_index = flow_past_cylinder(configuration)
if pde == 'Wave':
    fields, X, Y, dt, params = Wave_Spectral(configuration)

t = torch.arange(0, configuration['Data']['t_out']*dt, dt)
fields = fields[...,:configuration['Data']['t_out']]

XY = torch.stack((X,Y), dim=-1)
print("Fields shape: " + str(fields.shape))
print("Coords shape: " + str(XY.shape))

# %% 
# Normalising the data -- using the same normalisations for inputs and outputs
normalizer = nn.Identity()
fields_encoded = normalizer(fields)

from sklearn.model_selection import train_test_split
data = fields_encoded[...,:configuration['Data']['t_out']]
data_train, data_test = data[:int(configuration['Data']['ntrain']*(1-0.2))], data[-int(configuration['Data']['ntrain']*(0.2)):]
params_train, params_test = params[:int(configuration['Data']['ntrain']*(1-0.2))], params[-int(configuration['Data']['ntrain']*(0.2)):]

input_window = configuration['Train']['input_length']
prediction_steps = configuration['Train']['rollout_length'] - 1 

# --- 2. Initialize Dataset ---
train_dataset = GraphSpatioTemporalDataset(
    data=data_train,
    pos=XY,
    params=params_train,
    input_window=input_window,    
    prediction_steps=prediction_steps, 
    connectivity_radius=0.1
)

test_dataset = GraphSpatioTemporalDataset(
    data=data_test,
    pos=XY,
    params=params_test,
    input_window=input_window,     # Look at past 10 steps
    prediction_steps=configuration['Data']['t_out']-1,  # Predict next 1 step
    connectivity_radius=0.1
)

#Setting up the data loaders
from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=configuration['Data']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=configuration['Data']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)


# %%
from Models.GNNs import * 
if model_type == 'gcn':
    model = GCN(in_channels=1, hidden_channels=configuration['Model']['width'], out_channels=1, num_layers=configuration['Model']['depth'])
if model_type == 'nnconv':
    model = NNConvNet(in_channels=1, hidden_channels=configuration['Model']['width'], out_channels=1, num_layers=configuration['Model']['depth'], edge_dim=4)
model.to(device)

# print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning_rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler_step'], gamma=configuration['Opt']['scheduler_gamma'])
loss_func = torch.nn.MSELoss()

# %% 
#Training
start_time = default_timer()
epoch_init = 0
epochs = configuration['Opt']['epochs']
step, train_T_out, test_T_out = configuration['Data']['step'], configuration['Train']['rollout_length']-1, configuration['Data']['t_out']-1


for ep in tqdm(range(epoch_init, epochs+1)): 
    t1 = default_timer()
    train_loss = 0 
    for batch in train_loader:
        model.train()
        optimizer.zero_grad()
        pred = []
        for t in range(0, train_T_out, step):    
            if model_type == 'gcn':       
                im = model(batch.x.to(device), batch.edge_index.to(device))
            elif model_type == 'nnconv':
                im = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))
            # Compute loss for this step
            pred.append(im)
            # Update input for next timestep (sliding window)
            batch.x = im
        pred = torch.stack(pred, -1).squeeze(1)
        loss = loss_func(pred, batch.y.to(device))
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
        optimizer.step()
        train_loss += loss.item()

        # # Validation Loop
        # test_loss = 0
        # model.eval()
        # with torch.no_grad():
        #     for batch in test_loader:
        #         pred = []
        #         for t in range(0, test_T_out, step):        
        #             if model_type == 'gcn':       
        #                 out = model(batch.x.to(device), batch.edge_index.to(device))
        #             elif model_type == 'nnconv':
        #                 out = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))
                # pred.append(im)
        #             batch.x = out
        #         pred = torch.stack(pred, -1).squeeze(1)
        #         loss = loss_func(pred, batch.y.to(device)).item()#Computing loss for the full rollout. 

        #         test_loss += loss
    
    train_loss = train_loss / len(train_loader)
    # test_loss = test_loss / len(test_loader)
    t2 = default_timer()

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}") #, Test Loss: {round(test_loss,5)}")
    scheduler.step()

train_time = default_timer() - start_time

# %% 
#Evaluation
pred_set = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        loss = 0 
        pred = []
        for t in range(0, test_T_out, step):        
            if model_type == 'gcn':       
                out = model(batch.x.to(device), batch.edge_index.to(device))
            elif model_type == 'nnconv':
                out = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))
            batch.x = out
            # Collect predictions for full sequence loss
            pred.append(out)
        pred = torch.stack(pred, -1)

    # Split predictions by graph
        batch_size = batch.num_graphs
        for i in range(batch_size):
            mask = batch.batch == i
            graph_preds = pred[mask]
            pred_set.append(graph_preds)
    pred_set = torch.stack(pred_set, 0)

# %% 
#Plotting
#Shaping back to [BS, vars, Nt, Nx, Ny]
pred_set = pred_set.permute(0, 2, 3, 1)
pred_set = pred_set.reshape(pred_set.shape[0], pred_set.shape[1], pred_set.shape[2], 33, 33).cpu().numpy()

test_out = data_test[...,1:].permute(0, 1, 3, 2)
test_out = test_out.reshape(test_out.shape[0], test_out.shape[1], test_out.shape[2], 33, 33)

print(f"MSE: {np.mean(pred_set-test_out)**2}")
#%%
from Utils.plots import * 
class Run:
    def __init__(self, name=None):
        self.name = name

run = Run('test_'+model_type)
plots_2d_yaml(configuration, test_out, pred_set, plot_loc=os.getcwd(), run=run, idx=0, save=True)
# %% 
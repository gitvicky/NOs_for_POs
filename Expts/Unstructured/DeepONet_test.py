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
# %% 
#Loading config
def is_notebook():
    """Check if running in Jupyter notebook"""
    try:
        get_ipython()
        return True
    except NameError:
        return False

def load_config(config_path='configs/train/Wave_DeepONet.yaml'):
    """Load configuration from YAML file"""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        configuration = yaml.safe_load(f)
    print(f"Configuration loaded successfully!")
    return configuration

# Determine config path based on environment
if is_notebook():
    # Jupyter notebook mode - set your config path here
    CONFIG_PATH = '/pitagora/home/userexternal/vgopakum/NOs_for_POs/Expts/configs/train/Wave_DeepONet.yaml'  # Change this as needed
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

print("Data shape: " + str(fields.shape))

# %% 
# Normalising the data -- using the same normalisations for inputs and outputs
normalizer = nn.Identity()
fields_encoded = normalizer(fields)

from sklearn.model_selection import train_test_split
train_in, test_in, train_out, test_out = train_test_split(fields_encoded[...,:configuration['Data']['t_in']], fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']], test_size=configuration['Data']['test_train_split'], random_state=42)
params_train, params_test = params[:int(configuration['Data']['ntrain']*(1-0.2))], params[-int(configuration['Data']['ntrain']*(0.2)):]

train_data = torch.cat((train_in, train_out), dim=-1)#Merging for creating the windowed dataset.
input_window = configuration['Train']['input_length']
prediction_steps = configuration['Train']['rollout_length'] - 1 
train_dataset = SpatioTemporalDataset(train_data, params_train, input_window, prediction_steps)
test_dataset = DatasetWithParams(test_in, params_test, test_out)

#Setting up the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configuration['Data']['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configuration['Data']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
print("Training Input: " + str(train_in.shape))
print("Training Output: " + str(train_out.shape))

# %% 
#Model Definition 
# MIONet
from Models.DeepONet import * 
#Instantiating the Model
model = MIONet(
    in_channels=configuration['Model']['in_vars'],
    out_channels=configuration['Model']['out_vars'],
    branch_width = configuration['Model']['branch_width'],
    branch_depth = configuration['Model']['branch_depth'],
    trunk_width = configuration['Model']['trunk_width'],
    trunk_depth = configuration['Model']['trunk_depth'],
    x_in=X,
    y_in=Y)


# model = DON(
#     in_channels=configuration['Model']['in_vars'],
#     out_channels=configuration['Model']['out_vars'],
#     branch_width = configuration['Model']['branch_width'],
#     branch_depth = configuration['Model']['branch_depth'],
#     trunk_width = configuration['Model']['trunk_width'],
#     trunk_depth = configuration['Model']['trunk_depth'],
#     x_in=X,
#     y_in=Y)



model.to(device)
model.coords = model.coords.to(device)
print("Number of model params : " + str(model.count_params()))

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

# def forward(model, xx, dt):
#      return (model(xx[0]))

def forward(model, xx, dt): 
    """Euler method for temporal integration."""
    if dt == 0:
        raise ValueError("dt must be non-zero for Euler method")
    xx_new = xx[0] + model(xx) * dt 
    return xx_new 


for ep in tqdm(range(epoch_init, epochs+1)): 
    t1 = default_timer()
    train_loss = 0 
    for xx, yy in train_loader:
        model.train()
        optimizer.zero_grad()
        loss = 0 
        xx[0] = xx[0].to(device, non_blocking=True)
        xx[1] = xx[1].to(device, non_blocking=True)
        yy = yy.to(device, non_blocking=True)
        batch_size = xx[0].shape[0]

        for t in range(0, train_T_out, step):
            y = yy[..., t:t + step]
            
            im = forward(model, xx, dt)# Ensure output has time dimension

            # Compute loss for this step
            loss += loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
            
            # Collect predictions for full sequence loss
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            
            # Update input for next timestep (sliding window)
            xx[0] = torch.cat((xx[0][..., step:], im), dim=-1)
            print(xx[0].shape)
        
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
        optimizer.step()
        train_loss += loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()


        # Validation Loop
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for xx, yy in test_loader:
                xx[0], xx[1], yy = xx[0].to(device, non_blocking=True), xx[1].to(device, non_blocking=True), yy.to(device, non_blocking=True)
                batch_size = xx[0].shape[0]
                
                for t in range(0, test_T_out, step):
                    y = yy[..., t:t + step]
                    
                    out = forward(model, xx, dt)
                    
                    if t == 0:
                        pred = out
                    else:
                        pred = torch.cat((pred, out), -1)
                    
                    # Update input for next timestep
                    xx[0] = torch.cat((xx[0][..., step:], out), dim=-1)
                
                test_loss += loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
    
    train_loss = train_loss / len(train_loader)
    test_loss = test_loss / len(test_loader)
    t2 = default_timer()

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss,5)}")
    scheduler.step()

train_time = default_timer() - start_time

# %% 
#Evaluation

pred_set = []
model.eval()
with torch.no_grad():
    for xx, yy in tqdm(test_loader, desc="Running inference"):
        xx = [xx[0].to(device, non_blocking=True), 
            xx[1].to(device, non_blocking=True)]
        yy = yy.to(device, non_blocking=True)
        batch_size = xx[0].shape[0]
        pred_list = []
        
        for t in range(0, test_T_out, step):
            out = forward(model, xx, dt)
            

            pred_list.append(out)
            xx[0] = torch.cat((xx[0][..., step:], out), dim=-1)

        # Concatenate predictions
        if pred_list:
            pred = torch.cat(pred_list, dim=-1)
            pred_set.append(pred)

    if pred_set:
        pred_set = torch.cat(pred_set, dim=0)

        # Compute performance metrics
        test_out_device = test_out.to(device)

# %% 
from Expts.Unstructured.unstructured_plot import * 
fig, axes = create_field_comparison_plot(
X, Y, test_out.cpu(), pred_set.cpu(),
run=None,
batch_idx=15, var_idx=0,
time_steps=[0, 15, 30, 45],
title="Field: u",
obstacles=None,
test_label='Sim.',
pred_label='Net.'
)
plot_loc = os.getcwd()
plot_name = plot_loc + '/u_Field.png'
plt.savefig(plot_name, dpi=300, bbox_inches='tight')
# %%

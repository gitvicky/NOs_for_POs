#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Pipeline. 
"""
# %%
#Setting up simvue 
import os
import yaml 
import argparse

import sys
sys.path.append("..")
from Utils.simvue_utils import flatten_dict

#Config files.
def parse_args():
    parser = argparse.ArgumentParser(description='Training script with YAML config')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    return parser.parse_args()

args = parse_args()
with open(args.config, 'r') as f:
    configuration = yaml.safe_load(f)

run_config = flatten_dict(configuration)
# %% 
from simvue import Run, Client
with Run(mode='online') as run:

    run.init(folder=configuration['Simvue']['folder'], tags=['NPDE', configuration['Model']['arch'], 'POs4NOs', configuration['Physics']['pde'], configuration['Train']['odesolve']['method'], 'Tests'], metadata=run_config)

    #setting up the client API 
    client = Client()

    #Saving the current run file and the git hash of the repo
    run.save_file(os.path.abspath(__file__), 'code')
    run.save_file(os.path.abspath(args.config), 'code')

    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    run.update_metadata({'Git Hash': sha})

    #Importing the necessary packages
    import sys
    import numpy as np
    from tqdm import tqdm 
    import torch
    import torch.nn.functional as F
    from timeit import default_timer
    from tqdm import tqdm 

    #Setting up locations. 
    file_loc = os.getcwd()
    data_loc = os.path.dirname(os.getcwd()) + '/Data/'
    model_loc = file_loc + '/Weights/' + run.name
    os.mkdir(model_loc)
    plot_loc = file_loc + '/Plots'

    #Setting up the seeds and devices
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    # %%
    #Importing the models and utilities. 

    if configuration['Model']['arch'] == 'FNO':
        from Neural_PDE.Models.FNO import *
    elif configuration['Model']['arch'] == 'ViT':
        from Neural_PDE.Models.ViT_new import * 
    elif configuration['Model']['arch'] == 'U-Net':
        from Neural_PDE.Models.UNet import * 
    elif configuration['Model']['arch'] == 'CNO':
        from Neural_PDE.Models.CNO import * 
    elif configuration['Model']['arch'] == 'gMLP':
        from Neural_PDE.Models.gMLP_Vision import * 

    from Neural_PDE.Utils.processing_utils import * 
    from Neural_PDE.Utils.training_utils import * 

    # %% 
    ####################################
    # Data Preparation.
    ####################################

    t1 = default_timer()

    from data_loaders import *
    pde = configuration['Physics']['pde']
    if pde == 'Navier-Stokes':
        fields, x, y, dt = Navier_Stokes_Spectral(configuration['Data']['ntrain'])
    if pde == 'Incomp. Navier-Stokes':
        fields, force, x, y, dt = Navier_Stokes_Incomp(configuration['Data']['ntrain'])
    if pde == 'Comp. Navier-Stokes':
        fields, x, y, dt = Navier_Stokes_Comp(configuration['Data']['ntrain'], coeff=configuration['Physics']['coeff'])
    if pde == 'MHD':
        if configuration['Physics']['pde']['source'] == 'JOREK': 
            fields, x, y, dt = JOREK(configuration['Data']['ntrain'])
    
    t = torch.arange(0, fields.shape[-1], dt)

    fields = fields[...,:configuration['Data']['t_out']]

    #Making sure the data is in the correct format: [BS, N_vars, Nx, Ny, Nt]
    expected_shape = (configuration['Data']['ntrain'], configuration['Physics']['variables'], configuration['Physics']['Nx'], configuration['Physics']['Ny'], configuration['Data']['t_out'])
    assert fields.shape == expected_shape, \
        f"Expected fields shape to be {expected_shape}, but got {fields.shape}"

    # %%
    #Normalising the data -- using the same normalisations for inputs and outputs
    normalizer_func = Normalisation(configuration['Data']['normalisation'])
    normalizer = normalizer_func(fields)
    fields_encoded = normalizer.encode(fields)

    #Setting up train and test
    from sklearn.model_selection import train_test_split
    train_in, test_in, train_out, test_out = train_test_split(fields_encoded[...,:configuration['Data']['t_in']], fields_encoded[...,configuration['Data']['t_in']:configuration['Data']['t_out']], test_size=configuration['Data']['test-train-split'], random_state=42)
    print("Training Input: " + str(train_in.shape))
    print("Training Output: " + str(train_out.shape))

    #Saving Normalisation 
    saved_normalisations = model_loc + '/norms.npz'
    np.savez(saved_normalisations, 
            a=normalizer.a.numpy(), b=normalizer.b.numpy(), 
            )
    run.save_file(saved_normalisations, 'output')

    #Setting up the data loaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=configuration['Data']['batch size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=configuration['Data']['batch size'], shuffle=False)

    t2 = default_timer()
    print('preprocessing finished, time used:', t2-t1)

    # %% 
    ####################################
    # Setting up the Model and Optimizers 
    ####################################
    if configuration['Model']['operator splitting'] == True: 
    
    #With Operator Splitting.
        if pde == 'Navier-Stokes':
            from operator_splitting import NS_OS_rhs
            model = NS_OS_rhs(configuration)
        if pde == 'Incomp. Navier-Stokes':
            from operator_splitting import Incomp_NS_OS_rhs
            model = Incomp_NS_OS_rhs(configuration)
        if pde == 'Comp. Navier-Stokes':
            from operator_splitting import Comp_NS_OS_rhs
            model = Comp_NS_OS_rhs(configuration)
    
    else:
            
        if configuration['Model']['arch'] == 'FNO':
            if configuration['Model']['operator splitting'] == False:
                model = FNO_multi2d(configuration['Model']['in_vars'], 
                                    configuration['Model']['out_vars'], 
                                    configuration['Model']['modes'], 
                                    configuration['Model']['modes'],
                                    configuration['Model']['width'],
                                    configuration['Model']['n_layers']
                                    )
            
        if configuration['Model']['arch'] == 'U-Net':
            model = UNet2d(configuration['Data']['t_in'], 
                        configuration['Data']['step'], 
                        configuration['Model']['width'], 
                        configuration['Physics']['variables']
                        )
        
        if configuration['Model']['arch'] == 'ViT':
            model = ViT(
                image_size=(configuration['Physics']['Nx'], configuration['Physics']['Ny']),
                patch_size=(configuration['Model']['patch size'], configuration['Model']['patch size']),
                embed_dim=configuration['Model']['embed dim'],
                depth=configuration['Model']['depth'],
                n_heads=configuration['Model']['num heads'],
                channels=configuration['Physics']['variables'],
                mlp_dim = 256,
                dim_head = 32
                )
        
        if configuration['Model']['arch'] == 'CNO':
            model = CNO2d(in_dim = configuration['Model']['in channels'],             
                        out_dim = configuration['Model']['out channels'],
                        size = configuration['Model']['Nx'],
                        N_layers = configuration['Model']['N_layers'],
                        N_res = configuration['Model']['N_res'],
                        N_res_neck = configuration['Model']['N_res_neck'],
                        channel_multiplier = configuration['Model']['channel multiplier'],
                        use_bn = True
                        )      
            

        if configuration['Model']['arch'] == 'gMLP':
            model = gMLP(n_blocks = configuration['Model']['n_blocks'],
                        d_in = configuration['Model']['d_in'],
                        d_ffn = configuration['Model']['d_ffn'],
                        Nx = configuration['Model']['Nx'],
                        Ny = configuration['Model']['Ny'])


    model.to(device)
    run.update_metadata({'Number of Params': int(model.count_params())})
    print("Number of model params : " + str(model.count_params()))

    #Setting up the optimizer and scheduler, loss and epochs 
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler step'], gamma=configuration['Opt']['scheduler gamma'])
    if configuration['Model']['arch']=='fno':
        loss_func = LpLoss(size_average=False)
    else:
        loss_func = torch.nn.MSELoss()
    epoch_init = 0
    epochs = configuration['Opt']['epochs']

    #Restarting the run from a checkpoint 
    if configuration['Train']['restart'] == True: 
        client.get_artifact_as_file(client.get_artifact_as_file(configuration['Train']['restart run name']))
        ckpt_path = '/tmp/checkpoint.pt'
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch_init = checkpoint["epoch"]

    #Setting up the Training pipeline
    # if configuration['Train']['odesolve']['source'] == 'custom':
    from Utils import explicit_time
    train = explicit_time.Train_Setup(model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs,  ode_solver=configuration['Train']['odesolve']['source'], roll_out=configuration['Train']['odesolve']['method'])
    # elif configuration['Train']['odesolve']['source'] == 'torchdiffeq':
    #     from Utils import torch_odesolve
    #     train = torch_odesolve.Train_Setup(model, train_loader, test_loader, loss_func, optimizer, scheduler, epochs,  configuration['Train']['odesolve']['method'],  configuration['Train']['odesolve']['adjoint'])
    # %% 
    ####################################
    #Training
    ####################################
    start_time = default_timer()
    for ep in tqdm(range(epoch_init, epochs+1)): #Training Loop - Epochwise

        model.train()
        t1 = default_timer()
        train_loss, test_loss = train.one_epoch(configuration['Data']['step'], configuration['Data']['t_out']-1, dt=dt)
        t2 = default_timer()

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)

        print(f"Epoch {ep}, Time Taken: {round(t2-t1,3)}, Train Loss: {round(train_loss, 3)}, Test Loss: {round(test_loss,3)}")
        run.log_metrics({'Train Loss': train_loss, 'Test Loss': test_loss})
        
        # run.create_alert(
        #     name='Unstable',
        #     source='metrics',
        #     rule='is above',
        #     metric='Train Loss',
        #     frequency=1,
        #     window=1,
        #     threshold=1e5,
        #     trigger_abort=True
        #     )
                    
        scheduler.step()

        #Checkpointing. 
        if ep % configuration['Train']['checkpoint']['epochs'] == 0:
            checkpoint = {}
            checkpoint["model"] = model.state_dict()
            checkpoint["optimizer"] = optimizer.state_dict() 
            checkpoint["scheduler"] = scheduler.state_dict()
            checkpoint["epoch"] = ep
            torch.save(checkpoint, model_loc + "/checkpoint.pt")
            run.save_file(model_loc + "/checkpoint.pt", 'output')
            run.update_metadata({'Epochs': ep})

    train_time = default_timer() - start_time

    # %%
    # Saving the Model
    saved_model = model_loc + '/model.pth'
    torch.save( model.state_dict(), saved_model)
    run.save_file(saved_model, 'output')

    #Evaluation 
    # if configuration['Train']['odesolve']['source'] == 'custom':
    eval = explicit_time.Eval_Setup(model, test_in, test_out, normalizer='False', ode_solver = configuration['Train']['odesolve']['source'], roll_out= configuration['Train']['odesolve']['method'])

    # elif configuration['Train']['odesolve']['source'] == 'torchdiffeq':
    #     eval = torch_odesolve.Eval_Setup(model, test_in, test_out, roll_out= configuration['Train']['odesolve']['method'], ode_solver='torchdiffeq')
        
    pred_encoded, error = eval.inference(configuration['Data']['step'], configuration['Data']['t_out']-1, dt=dt)

    print('(MSE) Testing Error: %.3e' % (error))

    run.update_metadata({'Training Time': float(train_time),
                        'MSE Test Error': float(error)
                        })

    #Denormalising the test and predictions
    test_out = normalizer.decode(test_out.to(device)).cpu()
    pred_set = normalizer.decode(pred_encoded.to(device)).cpu()

    #Shaping back to [BS, vars, Nt, Nx, Ny]
    test_out = test_out.permute(0,1,4,2,3)
    pred_set = pred_set.permute(0,1,4,2,3)

    # %% 
    #Plotting the results 
    from Utils.plots import plots_2d_yaml
    idx = 0 
    plots_2d_yaml(configuration, test_out, pred_set, plot_loc, run, idx, save=True)
    # %%
    run.close()
    # %%

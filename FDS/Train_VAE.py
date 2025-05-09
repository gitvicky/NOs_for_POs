#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Pipeline for Conditional VAE with uncertainty visualization.
"""
# %%
# Setting up simvue 
import os
import yaml 
import argparse

import sys
sys.path.append("..")
from Utils.simvue_utils import flatten_dict
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# Config files
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

    run.init(folder=configuration['Simvue']['folder'], tags=[configuration['Physics']['pde'], configuration['Model']['arch']], metadata=run_config)

    # Setting up the client API 
    client = Client()

    # Saving the current run file and the git hash of the repo
    run.save_file(os.path.abspath(__file__), 'code')
    run.save_file(os.path.abspath(args.config), 'code')

    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    run.update_metadata({'Git Hash': sha})

    # Importing the necessary packages
    import sys
    import numpy as np
    from tqdm import tqdm 
    import torch
    import torch.nn.functional as F
    from timeit import default_timer
    from tqdm import tqdm 
    from sklearn.model_selection import train_test_split

    # Setting up locations
    file_loc = os.getcwd()
    data_loc = os.path.dirname(os.getcwd()) + '/Data/'
    model_loc = file_loc + '/Weights/' + run.name
    os.mkdir(model_loc)
    plot_loc = file_loc + '/Plots'

    # Setting up the seeds and devices
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    # %%
    # Importing the models and utilities
    from model_setup import *
    from Neural_PDE.Utils.processing_utils import * 
    from Neural_PDE.Utils.training_utils import * 
    from Cond_VAE import Conv3DVarConditionalAutoencoder, vae_loss_function

    # %% 
    ####################################
    # Data Preparation
    ####################################

    t1 = default_timer()

    from data_loaders import *
    pde = configuration['Physics']['pde']
    if pde == 'FDS':
        ins, conds, outs = FDS_Carpark(configuration)
        outs = outs.permute(0, 4, 1, 2, 3)

    # %%
    # Normalising the data -- using the same normalisations for inputs and outputs
    normalizer_func = Normalisation(configuration['Data']['normalisation'])
    normalizer_in = normalizer_func(ins)
    ins_encoded = normalizer_in.encode(ins)
    conds_encoded = normalizer_in.encode(conds)

    normalizer_out = normalizer_func(outs)
    outs_encoded = normalizer_out.encode(outs)

    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data import TensorDataset
    train_conds, test_conds, train_outs, test_outs = train_test_split(conds_encoded, outs_encoded, test_size=configuration['Data']['test-train-split'], random_state=42)

    train_loader = DataLoader(TensorDataset(train_conds, train_outs), batch_size=configuration['Data']['batch size'], shuffle=True) 
    test_loader = DataLoader(TensorDataset(test_conds, test_outs), batch_size=configuration['Data']['batch size'], shuffle=False)

    t2 = default_timer()
    print('preprocessing finished, time used:', t2-t1)

    # %% 
    ####################################
    # Setting up the Model and Optimizers 
    ####################################

    # VAE-specific model initialization
    if 'latent_dim' in configuration['Model']:
        latent_dim = configuration['Model']['latent_dim']
    else:
        latent_dim = 128  # Default value
        
    if 'beta' in configuration['Model']:
        beta = configuration['Model']['beta']  # KL weight for VAE
    else:
        beta = 1.0  # Default value
    
    # Number of samples for uncertainty estimation
    if 'n_samples' in configuration['Model']:
        n_samples = configuration['Model']['n_samples']
    else:
        n_samples = 10  # Default value

    # Initialize the VAE model
    model = Conv3DVarConditionalAutoencoder(
        in_channels=3,  # Assuming from your data
        out_channels=1,  # Assuming from your data
        conditional_features=conds_encoded.shape[1],
        latent_dim=latent_dim
    )
    
    model.to(device)

    run.update_metadata({'Number of Params': model.count_params()})
    print("Number of model params : " + str(model.count_params()))

    # Setting up the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Opt']['learning rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Opt']['scheduler step'], gamma=configuration['Opt']['scheduler gamma'])

    epoch_init = 0
    epochs = configuration['Opt']['epochs']

    # %% 
    ####################################
    # Training
    ####################################
    start_time = default_timer()

    def train_one_epoch():
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        for xx, yy in train_loader:
            xx, yy = xx.to(device), yy.to(device)
            grid = ins_encoded.unsqueeze(0).repeat(xx.shape[0], 1, 1, 1, 1).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through VAE
            recon, mu, logvar = model(grid, xx)
            print(recon.shape, mu.shape, logvar.shape, xx.shape, yy.shape)
            
            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss_function(recon, yy, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

        # Validation Loop
        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0
        
        with torch.no_grad():
            for xx, yy in test_loader:
                xx, yy = xx.to(device), yy.to(device)
                grid = ins_encoded.unsqueeze(0).repeat(xx.shape[0], 1, 1, 1, 1).to(device)
                
                # Forward pass
                recon, mu, logvar = model(grid, xx)
                
                # Calculate loss
                loss, recon_loss, kl_loss = vae_loss_function(recon, yy, mu, logvar, beta=beta)
                
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()

        # Average losses
        train_loss = train_loss / len(train_loader)
        train_recon_loss = train_recon_loss / len(train_loader)
        train_kl_loss = train_kl_loss / len(train_loader)
        
        test_loss = test_loss / len(test_loader)
        test_recon_loss = test_recon_loss / len(test_loader)
        test_kl_loss = test_kl_loss / len(test_loader)

        return train_loss, train_recon_loss, train_kl_loss, test_loss, test_recon_loss, test_kl_loss

    def validation(model, xx, yy):
        with torch.no_grad():
            xx, yy = xx.to(device), yy.to(device)
            grid = ins_encoded.unsqueeze(0).repeat(xx.shape[0], 1, 1, 1, 1).to(device)
            
            # Forward pass through VAE
            pred, mu, logvar = model(grid, xx)

            # Performance Metrics
            MSE_error = (yy - pred).pow(2).mean()
            MAE_error = torch.abs(yy - pred).mean()

        return pred, MSE_error, MAE_error
    
    def generate_samples_with_uncertainty(model, cond_features, grid, num_samples=10):
        """
        Generate multiple samples from the VAE to estimate uncertainty
        
        Args:
            model: VAE model
            cond_features: Conditional features
            grid: The grid input
            num_samples: Number of samples to generate
            
        Returns:
            mean_prediction: Mean of all samples
            std_prediction: Standard deviation of all samples
            samples: All generated samples
        """
        model.eval()
        samples_list = []
        
        with torch.no_grad():
            # Get the shape of what we'll be generating
            orig_shape = grid.shape
            batch_size = orig_shape[0]
            
            # Encode once to get mu and logvar
            mu, logvar = model.encode(grid)
            
            # Generate multiple samples from the latent distribution
            for i in range(num_samples):
                # Sample from latent distribution
                z = model.reparameterize(mu, logvar)
                
                # Decode to get a sample
                sample = model.decode(z, cond_features, orig_shape)
                samples_list.append(sample)
            
            # Stack all samples
            samples = torch.stack(samples_list, dim=0)  # [num_samples, batch_size, channels, ...]
            
            # Calculate mean and std across samples
            mean_prediction = torch.mean(samples, dim=0)
            std_prediction = torch.std(samples, dim=0)
            
        return mean_prediction, std_prediction, samples

    # Generate random samples function
    def generate_samples(model, num_samples=4):
        with torch.no_grad():
            # Use some test conditional features
            cond_features = test_conds[:num_samples].to(device)
            
            # Use the input shape from test data
            orig_shape = (num_samples, 1, outs.shape[2], outs.shape[3], outs.shape[4])
            
            # Generate samples
            samples = model.sample(num_samples, cond_features, orig_shape)
            
            # Denormalize the samples
            samples_denorm = normalizer_out.decode(samples)
            
        return samples_denorm

    for ep in tqdm(range(epoch_init, epochs+1)):  # Training Loop - Epochwise
        model.train()
        t1 = default_timer()
        
        train_loss, train_recon_loss, train_kl_loss, test_loss, test_recon_loss, test_kl_loss = train_one_epoch()
        
        t2 = default_timer()

        print(f"Epoch {ep}, Time: {round(t2-t1,3)}, Train Loss: {round(train_loss, 5)}, "
              f"Recon Loss: {round(train_recon_loss, 5)}, KL Loss: {round(train_kl_loss, 5)}, "
              f"Test Loss: {round(test_loss, 5)}")
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics to simvue
        run.log_metrics({
            'Train Loss': train_loss,
            'Train Reconstruction Loss': train_recon_loss,
            'Train KL Loss': train_kl_loss,
            'Test Loss': test_loss,
            'Test Reconstruction Loss': test_recon_loss,
            'Test KL Loss': test_kl_loss,
            'Learning Rate': current_lr
        }, step=ep)

        scheduler.step()

        # Checkpointing
        if ep % configuration['Train']['checkpoint']['epochs'] == 0:
            checkpoint = {}
            checkpoint["model"] = model.state_dict()
            checkpoint["optimizer"] = optimizer.state_dict() 
            checkpoint["scheduler"] = scheduler.state_dict()
            checkpoint["epoch"] = ep
            torch.save(checkpoint, model_loc + "/checkpoint_"+str(ep)+".pt")
            run.save_file(model_loc + "/checkpoint_"+str(ep)+".pt", 'output')
            run.update_metadata({'Epochs': ep})
            
            # Generate and save sample images at checkpoints
            if ep > 0:  # Skip first epoch
                samples = generate_samples(model)
                
                # Plot generated samples
                fig = plt.figure(figsize=plt.figaspect(0.5))
                for i in range(min(4, samples.shape[0])):
                    u_field = samples.detach().cpu()[i][0]
                    
                    for t in range(min(5, u_field.shape[2])):
                        ax = fig.add_subplot(4, 5, i*5 + t + 1)
                        pcm = ax.imshow(u_field[..., t], cmap=matplotlib.cm.coolwarm)
                        ax.title.set_text(f'Sample {i+1}, T{t+1}')
                        ax.axes.xaxis.set_ticks([])
                        ax.axes.yaxis.set_ticks([])
                        
                plt.tight_layout()
                sample_plot_name = plot_loc + '/VAE_samples_epoch_' + str(ep) + '_' + run.name + '.png'
                plt.savefig(sample_plot_name)
                run.save_file(sample_plot_name, 'output')
                plt.close()

    train_time = default_timer() - start_time

    # %%
    # Saving the Model
    saved_model = model_loc + '/vae_model.pth'
    torch.save(model.state_dict(), saved_model)
    run.save_file(saved_model, 'output')

    # Evaluation 
    pred_set_encoded, mse, mae = validation(model, test_conds, test_outs)

    print('(MSE) Testing Error: %.3e' % (mse))

    run.update_metadata({
        'Training Time': float(train_time),
        'MSE Test Error': float(mse),
        'Beta Value': float(beta),
        'Latent Dimension': int(latent_dim),
        'Uncertainty Samples': int(n_samples)
    })
    
    # Denormalising and reshaping the target and the predictions
    pred_set = normalizer_out.decode(pred_set_encoded.to(device)).cpu()
    test_u = normalizer_out.decode(test_outs.to(device)).cpu()
    
    # Generate samples from the trained model
    samples = generate_samples(model)
    
    # %%
    # Generate uncertainty visualization for a few test examples
    uncertainty_idx = [0, 1, 2]  # Visualize uncertainty for these test indices
    
    for idx in uncertainty_idx:
        # Select a single test example
        single_cond = test_conds[idx:idx+1].to(device)
        single_grid = ins_encoded.unsqueeze(0).repeat(1, 1, 1, 1, 1).to(device)
        single_target = test_u[idx:idx+1]
        
        # Generate multiple samples to estimate uncertainty
        mean_pred_encoded, std_pred_encoded, all_samples_encoded = generate_samples_with_uncertainty(
            model, single_cond, single_grid, num_samples=n_samples
        )
        
        # Denormalize predictions
        mean_pred = normalizer_out.decode(mean_pred_encoded.to(device)).cpu()
        
        # For standard deviation, we don't normalize/denormalize as it represents variation
        # But we scale it to be in the same units as the output
        # std_scaling_factor = normalizer_out.data_max - normalizer_out.data_min
        std_pred = std_pred_encoded.cpu() #* std_scaling_factor.cpu()
        
        # Denormalize all samples for visualization
        all_samples = []
        for i in range(all_samples_encoded.shape[0]):
            sample = normalizer_out.decode(all_samples_encoded[i:i+1].to(device)).cpu()
            all_samples.append(sample)
        
        # Create figure for this test example with uncertainty
        fig = plt.figure(figsize=(15, 12))
        
        # Plot target, mean prediction, and standard deviation
        rows = 3  # target, mean prediction, std deviation
        cols = 5  # time steps
        
        # Plot target
        for t in range(5):
            ax = fig.add_subplot(rows, cols, t+1)
            pcm = ax.imshow(single_target[0, 0, ..., t], cmap=matplotlib.cm.coolwarm)
            ax.set_title(f'Target T{t+1}')
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            if t == 0:
                plt.colorbar(pcm, ax=ax)
                
        # Plot mean prediction
        for t in range(5):
            ax = fig.add_subplot(rows, cols, cols+t+1)
            pcm = ax.imshow(mean_pred[0, 0, ..., t], cmap=matplotlib.cm.coolwarm)
            ax.set_title(f'Mean Prediction T{t+1}')
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            if t == 0:
                plt.colorbar(pcm, ax=ax)
                
        # Plot standard deviation (uncertainty)
        for t in range(5):
            ax = fig.add_subplot(rows, cols, 2*cols+t+1)
            pcm = ax.imshow(std_pred[0, 0, ..., t], cmap='viridis')
            ax.set_title(f'Uncertainty (Std Dev) T{t+1}')
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            if t == 0:
                plt.colorbar(pcm, ax=ax)
        
        plt.tight_layout()
        uncertainty_plot = plot_loc + f'/VAE_uncertainty_example_{idx}_{run.name}.png'
        plt.savefig(uncertainty_plot)
        run.save_file(uncertainty_plot, 'output')
        plt.close()
        
        # Plot individual samples to visualize distribution
        fig = plt.figure(figsize=(15, 15))
        
        # Plot a subset of the samples and the target for one time step
        time_step = 2  # Show uncertainty for the middle time step
        rows = 4  # 3x3 grid of samples + 1 for target
        cols = 3
        
        # Target
        ax = fig.add_subplot(rows, cols, 1)
        pcm = ax.imshow(single_target[0, 0, ..., time_step], cmap=matplotlib.cm.coolwarm)
        ax.set_title(f'Target T{time_step+1}')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.colorbar(pcm, ax=ax)
        
        # Mean prediction
        ax = fig.add_subplot(rows, cols, 2)
        pcm = ax.imshow(mean_pred[0, 0, ..., time_step], cmap=matplotlib.cm.coolwarm)
        ax.set_title(f'Mean Prediction T{time_step+1}')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.colorbar(pcm, ax=ax)
        
        # Standard deviation
        ax = fig.add_subplot(rows, cols, 3)
        pcm = ax.imshow(std_pred[0, 0, ..., time_step], cmap='viridis')
        ax.set_title(f'Uncertainty (Std Dev) T{time_step+1}')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.colorbar(pcm, ax=ax)
        
        # Plot individual samples (9 samples)
        n_viz_samples = min(9, n_samples)
        for i in range(n_viz_samples):
            ax = fig.add_subplot(rows, cols, i+4)
            pcm = ax.imshow(all_samples[i][0, 0, ..., time_step], cmap=matplotlib.cm.coolwarm)
            ax.set_title(f'Sample {i+1} T{time_step+1}')
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            
        plt.tight_layout()
        sample_viz_plot = plot_loc + f'/VAE_sample_distribution_{idx}_time{time_step}_{run.name}.png'
        plt.savefig(sample_viz_plot)
        run.save_file(sample_viz_plot, 'output')
        plt.close()
        
    # %%
    # Plot standard reconstruction comparison for evaluation
    idx = 0
    names = ['targets', 'reconstructions']
    for ii, u_field in enumerate([test_u, pred_set]):
        u_field = u_field[idx][0]
            
        v_min = torch.min(u_field)
        v_max = torch.max(u_field)

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 5, 1)
        pcm = ax.imshow(u_field[...,0], cmap=matplotlib.cm.coolwarm)
        ax.title.set_text('T1')
        fig.colorbar(pcm, pad=0.05)

        u_field = u_field
        ax = fig.add_subplot(1, 5, 2)
        pcm = ax.imshow(u_field[...,1], cmap=matplotlib.cm.coolwarm)
        ax.title.set_text('T2')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(1, 5, 3)
        pcm = ax.imshow(u_field[...,2], cmap=matplotlib.cm.coolwarm)
        ax.title.set_text('T3')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(1, 5, 4)
        pcm = ax.imshow(u_field[...,3], cmap=matplotlib.cm.coolwarm)
        ax.title.set_text('T4')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(1, 5, 5)
        pcm = ax.imshow(u_field[...,4], cmap=matplotlib.cm.coolwarm)
        ax.title.set_text('T5')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        plot_name = plot_loc + '/VAE_T_'+ names[ii] + '_' + run.name + '.png'
        plt.savefig(plot_name)
        run.save_file(plot_name, 'output')
        plt.close()
    
    # Generate latent space visualization if 2D latent space
    if latent_dim == 2:
        # Collect latent representations
        latent_points = []
        condition_values = []
        
        with torch.no_grad():
            for xx, yy in test_loader:
                xx, yy = xx.to(device), yy.to(device)
                grid = ins_encoded.unsqueeze(0).repeat(xx.shape[0], 1, 1, 1, 1).to(device)
                
                # Get latent representation (mu)
                mu, _ = model.encode(grid)
                
                latent_points.append(mu.cpu().numpy())
                condition_values.append(xx.cpu().numpy())
        
        latent_points = np.vstack(latent_points)
        condition_values = np.vstack(condition_values)
        
        # Plot 2D latent space
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_points[:, 0], latent_points[:, 1], c=condition_values[:, 0], cmap='viridis', alpha=0.7)
        plt.colorbar(label='Condition Value')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('VAE Latent Space Visualization')
        
        latent_plot_name = plot_loc + '/VAE_latent_space_' + run.name + '.png'
        plt.savefig(latent_plot_name)
        run.save_file(latent_plot_name, 'output')

    run.close()
    # %%
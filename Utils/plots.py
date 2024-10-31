import torch 
import matplotlib
from matplotlib import pyplot as plt


def plots_2d(configuration, test_out, pred_set, plot_loc, run, idx=0, save=True):
    field = configuration['Field'].split(',')
    field = [letter.strip() for letter in field]
    for var in range(configuration['Variables']):
        u_field = test_out[idx][var]
            
        v_min_1 = torch.min(u_field[0])
        v_max_1 = torch.max(u_field[0])

        v_min_2 = torch.min(u_field[configuration['T_out'] // 2])
        v_max_2 = torch.max(u_field[configuration['T_out'] // 2])

        v_min_3 = torch.min(u_field[-1])
        v_max_3 = torch.max(u_field[-1])

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(2, 3, 1)
        pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm, vmin=v_min_1, vmax=v_max_1)
        # ax.title.set_text('Initial')
        ax.title.set_text('t=' + str(configuration['T_in']))
        ax.set_ylabel('Solution -  ' + field[var])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 2)
        pcm = ax.imshow(u_field[configuration['T_out'] // 2], cmap=matplotlib.cm.coolwarm, vmin=v_min_2,
                        vmax=v_max_2)
        # ax.title.set_text('Middle')
        ax.title.set_text('t=' + str((configuration['T_out']+ configuration['T_in']) // 2))
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 3)
        pcm = ax.imshow(u_field[ -1], cmap=matplotlib.cm.coolwarm, vmin=v_min_3, vmax=v_max_3)
        # ax.title.set_text('Final')
        ax.title.set_text('t=' + str(configuration['T_out']))
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        u_field = pred_set[idx][var]

        ax = fig.add_subplot(2, 3, 4)
        pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm, vmin=v_min_1, vmax=v_max_1)
        ax.set_ylabel(configuration['Model'])

        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 5)
        pcm = ax.imshow(u_field[int(configuration['T_out']/ 2)], cmap=matplotlib.cm.coolwarm, vmin=v_min_2,
                        vmax=v_max_2)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 6)
        pcm = ax.imshow(u_field[-1], cmap=matplotlib.cm.coolwarm, vmin=v_min_3, vmax=v_max_3)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        if save == True:
            plot_name = plot_loc + '/' + field[var] + '_' + run.name + '.png'
            plt.savefig(plot_name)
            run.save_file(plot_name, 'output')
            

def plots_2d_yaml(configuration, test_out, pred_set, plot_loc, run, idx=0, save=True):
    field = configuration['Physics']['field'].split(',')
    field = [letter.strip() for letter in field]
    for var in range(configuration['Physics']['variables']):
        u_field = test_out[idx][var]
            
        v_min_1 = torch.min(u_field[0])
        v_max_1 = torch.max(u_field[0])

        v_min_2 = torch.min(u_field[configuration['Data']['t_out'] // 2])
        v_max_2 = torch.max(u_field[configuration['Data']['t_out'] // 2])

        v_min_3 = torch.min(u_field[-1])
        v_max_3 = torch.max(u_field[-1])

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(2, 3, 1)
        pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm, vmin=v_min_1, vmax=v_max_1)
        # ax.title.set_text('Initial')
        ax.title.set_text('t=' + str(configuration['Data']['t_in']))
        ax.set_ylabel('Solution -  ' + field[var])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 2)
        pcm = ax.imshow(u_field[configuration['Data']['t_out'] // 2], cmap=matplotlib.cm.coolwarm, vmin=v_min_2,
                        vmax=v_max_2)
        # ax.title.set_text('Middle')
        ax.title.set_text('t=' + str((configuration['Data']['t_out']+ configuration['Data']['t_in']) // 2))
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 3)
        pcm = ax.imshow(u_field[ -1], cmap=matplotlib.cm.coolwarm, vmin=v_min_3, vmax=v_max_3)
        # ax.title.set_text('Final')
        ax.title.set_text('t=' + str(configuration['Data']['t_out']))
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        u_field = pred_set[idx][var]

        ax = fig.add_subplot(2, 3, 4)
        pcm = ax.imshow(u_field[0], cmap=matplotlib.cm.coolwarm, vmin=v_min_1, vmax=v_max_1)
        ax.set_ylabel('Prediction')

        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 5)
        pcm = ax.imshow(u_field[int(configuration['Data']['t_out']/ 2)], cmap=matplotlib.cm.coolwarm, vmin=v_min_2,
                        vmax=v_max_2)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        ax = fig.add_subplot(2, 3, 6)
        pcm = ax.imshow(u_field[-1], cmap=matplotlib.cm.coolwarm, vmin=v_min_3, vmax=v_max_3)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        fig.colorbar(pcm, pad=0.05)

        if save == True:
            plot_name = plot_loc + '/' + field[var] + '_' + run.name + '.png'
            plt.savefig(plot_name)
            run.save_file(plot_name, 'output')
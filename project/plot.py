from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as tik
import os
from matplotlib import cm
from neural import NeuralState


def plot_weigth_matrix_bars(m: np.ndarray):
    """
    Plot a weight matrix as 3d bar diagram
    :param m: Weight matrix
    :return: -
    """

    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create an X-Y mesh of the same dimension as the 2D data
    x_s, y_s = np.meshgrid(np.arange(m.shape[1]), np.arange(m.shape[0]))

    x_s = x_s.flatten()
    y_s = y_s.flatten()
    z_data = m.flatten()

    ax.bar(x_s, y_s, zs=z_data, zdir='y', alpha=0.8)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('Weight')
    
    plt.show()


def hinton(matrix: np.ndarray, file: str = "", max_weight=None):
    """
    Draw Hinton diagram for visualizing a weight matrix.
    :param matrix: Input 2D matrix.
    :param file: File path for saving the plot.
    :param max_weight: Manually set upper limit for values.
    :return: Shows the Hinton diagram as new window or saves it to a file.
    """
    ax = plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('none')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    if file == "":
        plt.show()
    else:
        plt.savefig(file)
    plt.close()


def height_plot(matrix: np.ndarray, file: str = ""):
    """
    Draw temperature height map diagram.
    :param matrix: Input 2D matrix.
    :param file: File path for saving the plot.
    :return: Shows the height map diagram as new window or saves it to a file.
    """
    
    # Create heights in the grid
    z = matrix
    
    # Build a figure with 2 subplots, the first is 3D
    fig = plt.figure()
    
    ax2 = fig.add_subplot(111)
    
    im = ax2.imshow(z, cmap="hot", interpolation='none')
    ax2.invert_yaxis()
    
    # add an explanatory colour bar
    plt.colorbar(im, orientation='vertical')
    
    if file == "":
        plt.show()
    else:
        plt.savefig(file)
    plt.close()


def combined_plot1(weights: list, times: list, dweights: list, stepsize: int,
                   neurons: np.ndarray, hopfield: np.ndarray, file: str = None, metadata: str = ""):
    """
    
    :param weights:
    :param times:
    :param dweights:
    :param stepsize:
    :param neurons:
    :param hopfield:
    :param file:
    :param metadata:
    :return:
    """
    
    l = len(weights)
    
    w = weights[0::stepsize]
    c_w = len(w)
    dw = [sum(dweights[i:i+stepsize]) for i in range(0, l - 1, stepsize)]
    c_dw = len(dw)
    
    l_ax = max(4, c_w + 1)

    # Build a figure with 2 subplots, the first is 3D
    fig, axes = plt.subplots(ncols=l_ax, nrows=4)
    size = 5
    fig.set_size_inches(l_ax * size, 3 * size)

    #
    # Title

    fig.suptitle(metadata, fontsize=14, fontweight='bold')
    
    for i in range(2, l_ax - 2):
        fig.delaxes(axes[0][i])
    
    #
    # Neuron Map

    major_locator_n = tik.MultipleLocator(neurons.shape[0] // 2)
    major_formatter_n = tik.FormatStrFormatter('%d')
    minor_locator_n = tik.MultipleLocator(1)
    
    ax = axes[0][-1]
    z = neurons
    im = ax.imshow(z, cmap="hot", interpolation='none')
    ax.set_aspect('equal')
    ax.set_title("Active Neurons")

    ax.yaxis.set_major_locator(major_locator_n)
    ax.yaxis.set_major_formatter(major_formatter_n)
    ax.yaxis.set_minor_locator(minor_locator_n)

    ax.xaxis.set_major_locator(major_locator_n)
    ax.xaxis.set_major_formatter(major_formatter_n)
    ax.xaxis.set_minor_locator(minor_locator_n)

    ax = axes[0][-2]
    ax.set_aspect(8)
    fig.colorbar(im, orientation='vertical', cax=ax)

    #
    # Hopfield
    
    major_locator_w = tik.MultipleLocator(hopfield.shape[0] // 2)
    major_formatter_w = tik.FormatStrFormatter('%d')
    minor_locator_w = tik.MultipleLocator(hopfield.shape[0] // 4)
    
    ax = axes[0][0]
    z = hopfield
    im = ax.imshow(z, cmap="hot", interpolation='none')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title("Hopfield weights")
    ax.yaxis.tick_right()
    
    ax.yaxis.set_major_locator(major_locator_w)
    ax.yaxis.set_major_formatter(major_formatter_w)
    ax.yaxis.set_minor_locator(minor_locator_w)

    ax.xaxis.set_major_locator(major_locator_w)
    ax.xaxis.set_major_formatter(major_formatter_w)
    ax.xaxis.set_minor_locator(minor_locator_w)
    
    ax = axes[0][1]
    ax.set_aspect(8)
    fig.colorbar(im, orientation='vertical', cax=ax)
    ax.yaxis.tick_left()
        
    #
    # Weights & Weights per neuron

    weight_min = np.min(w)
    weight_max = np.max(w)
    
    for i in range(c_w):
        ax = axes[1][i]
        z = w[i]
        
        im = ax.imshow(z, cmap="hot", interpolation='none', vmin=weight_min, vmax=weight_max)
        
        ax.invert_yaxis()
        ax.set_aspect('equal')
        if i == 0:
            ax.yaxis.set_major_locator(major_locator_w)
            ax.yaxis.set_major_formatter(major_formatter_w)
            ax.yaxis.set_minor_locator(minor_locator_w)

            ax.xaxis.set_major_locator(major_locator_w)
            ax.xaxis.set_major_formatter(major_formatter_w)
            ax.xaxis.set_minor_locator(minor_locator_w)
            ax.set_title("Weights: t = " + '% 4.2f' % times[i * stepsize])
        else:
            ax.set_axis_off()
            ax.set_title("t = " + '% 4.2f' % times[i * stepsize])

        ax = axes[3][i]
        weight_per_neuron(ax, z, neurons.flatten())
        
        if i != 0:
            ax.set_axis_off()
        else:
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_title("Weight per neuron (colored: only active):")
        
    ax = axes[1][-1]
    
    ax.set_aspect(8)
    fig.colorbar(im, orientation='vertical', cax=ax, extend='both')

    fig.delaxes(axes[3][-1])
    
    #
    # dWeights
    
    dweight_min = np.min(dw)
    dweight_max = np.max(dw)

    for i in range(c_dw):
        ax = axes[2][i]
        z = dw[i]
    
        im = ax.imshow(z, cmap="hot", interpolation='none', vmin=dweight_min, vmax=dweight_max)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        if i == 0:
            ax.yaxis.set_major_locator(major_locator_w)
            ax.yaxis.set_major_formatter(major_formatter_w)
            ax.yaxis.set_minor_locator(minor_locator_w)

            ax.xaxis.set_major_locator(major_locator_w)
            ax.xaxis.set_major_formatter(major_formatter_w)
            ax.xaxis.set_minor_locator(minor_locator_w)
            ax.set_title("Deviations:")
        else:
            ax.set_axis_off()
    
    fig.delaxes(axes[2][-2])

    ax = axes[2][-1]
    ax.set_aspect(8)
    fig.colorbar(im, orientation='vertical', cax=ax, extend='both')
    
    #
    # Finish

    fig.tight_layout()
    
    if not file:
        plt.show()
    else:
        i = 0
        while os.path.exists('{}_{:d}.png'.format(file, i)):
            i += 1
        file = '{}_{:d}.png'.format(file, i)
        print("Saving results to: " + file)
        plt.savefig(file, dpi=100)
        
    plt.close()


def combined_learning_plot_patternwise(weights: list, times: list, dweights: list, neurons_t: list, neuralstates: list,
                                       spp: int, rot: int, file: str = None):
    c_pat = len(neuralstates)
    l_ax = c_pat + 2
    
    w = weights[0::spp]
    t = times[0::spp]
    n = neurons_t[0::spp]
    metadata = ""
    
    #
    # Prepare plot
    
    fig, axes = plt.subplots(ncols=l_ax, nrows=3)
    size = 5
    fig.set_size_inches(l_ax * size, 3 * size)
    
    #
    # Title
    
    ax = axes[0][0]
    ax.set_title(metadata, fontsize=14, fontweight='bold')
    ax.set_axis_off()
    
    #
    # Plots

    state_0 = neuralstates[0]

    weight_min = np.min(w)
    weight_max = np.max(w)

    major_locator_w = tik.MultipleLocator(state_0.N // 2)
    major_formatter_w = tik.FormatStrFormatter('%d')
    minor_locator_w = tik.MultipleLocator(state_0.N // 4)
    
    for i in range(l_ax - 1):
        #
        # Neuron Map
        
        if 0 < i < len(n) + 1:
            ax = axes[0][i]
            state = n[i-1]
            
            z = state.as_matrix()
            
            if i == 1:
                neural_map(ax, z, True)
                ax.set_title("Active Neurons")
            else:
                neural_map(ax, z, False)
    
        #
        # Weights
        ax_w = axes[1][i]
        z = w[i]

        im_w = ax_w.imshow(z, cmap="hot", interpolation='none', vmin=weight_min, vmax=weight_max)

        ax_w.invert_yaxis()
        ax_w.set_aspect('equal')
        if i == 0:
            ax_w.yaxis.set_major_locator(major_locator_w)
            ax_w.yaxis.set_major_formatter(major_formatter_w)
            ax_w.yaxis.set_minor_locator(minor_locator_w)

            ax_w.xaxis.set_major_locator(major_locator_w)
            ax_w.xaxis.set_major_formatter(major_formatter_w)
            ax_w.xaxis.set_minor_locator(minor_locator_w)
            ax_w.set_title("Weights: t = " + '% 4.2f' % 0)
        else:
            ax_w.set_axis_off()
            ax_w.set_title("t = " + '% 4.2f' % t[i])
            
        #
        # Weights per neuron
        ax = axes[2][i]
        if i == 0:
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_title("Weight per neuron (colored: only active):")
            wpn_n = np.zeros(state_0.N)
        else:
            ax.set_axis_off()
            wpn_n = state.vec

        weight_per_neuron(ax, z, wpn_n)

        #
        # Colorbar
        if i == l_ax - 2:
            ax = axes[1][-1]
            ax.set_aspect(8)
        
            fig.colorbar(im_w, orientation='vertical', cax=ax, extend='both')

    #
    # Empty axes
    ax = axes[0][-1]
    fig.delaxes(ax)
    
    ax = axes[2][-1]
    fig.delaxes(ax)
        
    #
    # Finish
    
    fig.tight_layout()
    
    if not file:
        plt.show()
    else:
        i = 0
        while os.path.exists('{}_{:d}.png'.format(file, i)):
            i += 1
        file = '{}_{:d}.png'.format(file, i)
        print("Saving results to: " + file)
        plt.savefig(file, dpi=100)
    
    plt.close()


def weight_per_neuron(ax: plt.Axes, w: np.ndarray, neurons: np.ndarray):

    width = 0.7
    num = w.shape[0]
    
    w_n, w_n_a, x_n_a = [], [], []

    x_n = np.arange(1, num + 1)
    
    for i in range(num):
        w_n.append(np.sum(w[i]))
        
        if neurons[i] == 1:
            sm = 0
            for j in range(num):
                sm += w[i][j] if neurons[j] == 1 else 0
            w_n_a.append(sm)
            x_n_a.append(x_n[i])
        
    w_max = np.max(w_n)

    # customize layout
    step = (num // 10)
    steps = x_n[0::max(1, step)]
    steps = np.array(steps) - 1
    steps[0] = 1
    if steps[-1] != x_n[-1]:
        steps = np.append(steps, x_n[-1])
    
    major_locator_n = tik.FixedLocator(steps)
    major_locator_n.view_limits(1, num)
    minor_locator_n = tik.MultipleLocator(1)
    ax.xaxis.set_major_locator(major_locator_n)
    ax.xaxis.set_minor_locator(minor_locator_n)

    ax.set_xlim(0, num + 1)
    ax.set_ylim(0, max(2, w_max))

    # colormap for active neurons:
    y = np.array(w_n_a) - 1
    sp = cm.get_cmap("spring").reversed()
    atu = cm.get_cmap("autumn").reversed()
    colors = [atu(abs(y_i) / 1) if y_i < 0 else sp(y_i / max(1, w_max - 1)) for y_i in y]

    # red dash line:
    ax.plot((0, num + 1), (1, 1), 'red', linestyle='--')
    
    # gray bars for inactive neurons
    ax.bar(x_n, w_n, width, color='gray')
    
    # colored active neurons
    ax.bar(x_n_a, w_n_a, width, color=colors)


def neural_map(ax: plt.Axes, neurons: np.ndarray, axes: bool):
    l = neurons.shape[0]
    if axes:
        major_locator_n = tik.MultipleLocator(l // 2)
        major_formatter_n = tik.FormatStrFormatter('%d')
        minor_locator_n = tik.MultipleLocator(1)
        
        ax.yaxis.set_major_locator(major_locator_n)
        ax.yaxis.set_major_formatter(major_formatter_n)
        ax.yaxis.set_minor_locator(minor_locator_n)
        
        ax.xaxis.set_major_locator(major_locator_n)
        ax.xaxis.set_major_formatter(major_formatter_n)
        ax.xaxis.set_minor_locator(minor_locator_n)
    else:
        ax.xaxis.set_major_locator(tik.NullLocator())
        ax.xaxis.set_minor_locator(tik.NullLocator())
        ax.yaxis.set_major_locator(tik.NullLocator())
        ax.yaxis.set_minor_locator(tik.NullLocator())
    
    ax.imshow(neurons, cmap="hot", interpolation='none')
    ax.set_aspect('equal')

    ma = l - 0.5
    mi = -0.5
    ax.set_xlim(mi, ma)
    ax.set_ylim(mi, ma)
    
    for i in range(1, l):
        xy = i - 0.5
        ax.plot((mi, ma), (xy, xy), 'red', linestyle='-')
        ax.plot((xy, xy), (mi, ma), 'red', linestyle='-')

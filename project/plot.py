from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


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


def height_plot(matrix: np.ndarray, file: str = "", ):
    """
    Draw temperature height map diagram.
    :param matrix: Input 2D matrix.
    :param file: File path for saving the plot.
    :return: Shows the height map diagram as new window or saves it to a file.
    """
    # xr = np.arange(matrix.shape[0])
    # yr = np.arange(matrix.shape[1])
    # x, y = np.meshgrid(xr, yr)
    
    # Create heights in the grid
    z = matrix
    
    # Build a figure with 2 subplots, the first is 3D
    fig = plt.figure()
    # ax = fig.add_subplot(2, 1, 1, projection='3d')
    # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    
    ax2 = fig.add_subplot(111)
    im = ax2.imshow(z, cmap='hot', interpolation='none')
    # swap the Y axis so it aligns with the 3D plot
    ax2.invert_yaxis()
    
    # add an explanatory colour bar
    plt.colorbar(im, orientation='vertical')
    
    if file == "":
        plt.show()
    else:
        plt.savefig(file)
    plt.close()

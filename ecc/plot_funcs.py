import os
import pandas as pd
import numpy as np
import scipy.ndimage as ndi
from matplotlib import pyplot as plt

from easy_cell_counter import image_utils as iut

##### PLOT STYLES #####
plt.style.use('seaborn-white')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def plot_intensity_profile(table_path, nickname=None,
                           col_name='deltaI', savename=None, intensity_range=None):
    """
    using the detected cell table, plot the intensity profile
    Inputs:
        table_path: path to cell table (csv)
    Optional Inputs:
        nickname: this will be used in the plot title. Default: None
        col_name: the column name to use to generate a plot. Default: 'DeltaI'
        savename: If specified, the plot is saved with the given file name.
                  It should end with compatible file extension supported by matplotlib.
                  Default: None
        intensity_range: controls the range of x-axis, should be a tuple of length 2
                         e.g. intensity_range=(0, 65535)
                         default: None (determined automatically)
    """
    table = pd.read_csv(table_path, header=0)

    intensity = table[col_name].values
    if intensity_range is None:
        intensity_range = (intensity.min(), intensity.max())

    hist, bin_edges = np.histogram(intensity, bins=256,
                                   range=intensity_range,
                                   density=True)
    bins = ( bin_edges[:-1] + bin_edges[1:] ) / 2

    f, axs = plt.subplots(1, 2, figsize=(10,5) )

    axs[0].plot( bins, hist ) # linear plot
    axs[1].semilogy( bins, hist ) # semilog plot
    for ax in axs:
        ax.set_xlabel('Intensity (AU)')
        ax.set_ylabel('Normalized count')
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        title = f'Intensity profile'
        if nickname is not None:
            title += f': {nickname}'
        ax.set_title(title)

    print("Max intensity:", intensity.max())
    print("Mean intensity:", intensity.mean())
    print("Peak intensity:", bins[hist.argmax()])

    # save plot
    if savename is not None:
        # make sure that directory is ready
        dirname = os.path.dirname(savename)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        # save figure
        plt.savefig(savename, dpi=300, 
                    transparent=True, bbox_inches='tight' )
        print(f"Plot saved as {savename}")

        # save raw values as csv
        savename = os.path.splitext(savename)[0] + '.csv'
        values = pd.DataFrame(columns=['intensity', 'normalized_count'])
        values['intensity'] = bins
        values['normalized_count'] = hist
        values.to_csv(savename, index=False)
        print(f"Plot raw values saved as {savename}")
    plt.show()

def plot_SBR( table_path, nickname, 
              savename=None, plot_x_range=None ):
    """
    using the detected cell table, plot the intensity profile
    INPUTS:
        table_path: path to cell table (csv)
        savename: if not None, the plot is saved as a PNG file with the given file name
        intensity_range: controls the range of x-axis, should be a tuple of length 2
                         e.g. intensity_range=(0, 65535)
    """
    table = pd.read_csv( table_path, header=0 )

    deltaI = table["deltaI"].values
    bg = table["BG"].values
    intensity = deltaI + bg
    # we do not want 0!
    bg[bg==0] = np.nan
    sbr = intensity / bg

    if plot_x_range is None:
        plot_x_range = (0, sbr.max() )

    hist, bin_edges = np.histogram( sbr, bins=256, 
                                    range=plot_x_range,
                                    density=True)
    bins = ( bin_edges[:-1] + bin_edges[1:] ) / 2

    f, axs = plt.subplots(1, 2, figsize=(10,5) )

    axs[0].plot( bins, hist )
    axs[1].semilogy( bins, hist )
    for ax in axs:
        ax.set_xlabel('SBR')
        ax.set_ylabel('Normalized count')
        ax.set_title('Profile of detected cells: ' + nickname)
        # horizontal grid line
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)

    print( "Mean SBR:", sbr.mean() )
    print( "Peak SBR:", bins[ hist.argmax() ] )
    print( "Min SBR:", sbr.min() )
    print( "5% percentile:", np.percentile( sbr, 5 ) )

    # save plot
    if not savename is None:
        plt.savefig( savename, dpi=300, 
                     transparent=True, bbox_inches='tight' )
        print("Plot saved as", savename)
        # save raw values as csv
        savename = os.path.splitext(savename)[0] + '.csv'
        values = pd.DataFrame( columns=['intensity', 'normalized_count'] )
        values['intensity'] = bins
        values['normalized_count'] = hist
        values.to_csv( savename, index=False)

    plt.show()

def plot_cell_positions(table_path, savename, imgshape, vx_size):
    """
    plot detected cell positions, and saves it as a image (NIFTI format)
    Inputs:
        table_path: path to cell table (csv)
        savename: Output file name. should end with .nii.gz
        imgshape: shape of the input image, order should follow that of python array (i.e. ZYX order)
        vx_size: python dictionary object, which defines the voxel spacing (in um) of the input image
                 e.g. vx_size = {"X": 6.45, "Y": 6.45, "Z": 7.0}
    """
    # prepare array to plot cells
    result = np.zeros(imgshape, dtype=np.uint8)
    
    # load table
    table = pd.read_csv(table_path, sep=',', header=0)

    # convert from um to pixels
    X = np.round(table['X'].values / vx_size["X"])
    Y = np.round(table['Y'].values / vx_size["Y"])
    Z = np.round(table['Z'].values / vx_size["Z"])

    # plot cell points
    for x,y,z in zip(X,Y,Z):
        x = int(x)
        y = int(y)
        z = int(z)
        result[z,y,x] = 1

    # for visibility, dilate the image
    struct = ndi.generate_binary_structure(3,1)
    result = ndi.filters.maximum_filter(result, footprint=struct)

    # write as a NIFTI image
    if savename.endswith(('.nii','.nii.gz')):
        iut.write_as_nifti(result, savename, 1,1,1)
        print("Plot saved as", savename)
    else:
        msg = "Invalid file extension! It should end with either .nii or .nii.gz"
        raise ValueError(msg)
from cloudmap import image_utils as iut
from cloudmap import registration

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.io as sio
from scipy import spatial
import os

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15


class AlignChannels:

    def __init__( self ):
        self.verbose = False
        self.nickname = 'untitled'
        self.fixed_scaling = 1.0
        self.moving_scaling = 1.0
        self.ants_path = None

    def set_verbose( self, TorF ):
        self.verbose = TorF

    def set_outdir( self, outdir ):
        # make sure that output directory is ready
        if not outdir.endswith('/'):
            warnings.warn("'outdir' must end with '/'", SyntaxWarning)
            outdir += '/'
        if not os.path.exists( outdir ):
            os.mkdir( outdir )
        self.outdir = outdir

    def set_nickname( self, nickname ):
        self.nickname = nickname

    def set_fixed_channel( self, img_path ):
        if not os.path.exists( img_path ):
            raise FileNotFoundError("Cannot find", img_path )
        self.fixed_path = img_path

    def set_fixed_scaling( self, scaling ):
        self.fixed_scaling = scaling

    def set_moving_scaling( self, scaling ):
        self.moving_scaling = scaling

    def set_ants_path( self, ants_path ):
        self.ants_path = ants_path

    def set_moving_channel( self, img_path ):
        if not os.path.exists( img_path ):
            raise FileNotFoundError("Cannot find", img_path )
        self.moving_path = img_path

    def set_crop_ranges( self, crop_ranges ):
        self.crop_ranges = crop_ranges

    def set_registration_params( self, paramfile ):
        self.registration_params = paramfile

    def crop_images( self ):

        # work on fixed and moving image
        for ch in [ "fixed", "moving"]:
            if ch == "fixed":
                h5path = self.fixed_path
                scaling = self.fixed_scaling
            else:
                h5path = self.moving_path
                scaling = self.moving_scaling
            for (k, cr) in enumerate( self.crop_ranges):
                filename = self.outdir + ch + '_{:02d}.tiff'.format(k)
                self._crop_hdf5( h5path, cr, filename, scaling )

    def _crop_hdf5( self, h5path, crop_range, filename, scaling=1 ):
        # load image
        img = iut.load_hdf5_image( h5path, datarange=crop_range )
        if self.verbose:
            print( "Image shape:", img.shape )

        # adjust intensity
        img = (scaling * img).astype(np.uint16)

        # write image
        iut.write_as_tiff( img, filename, bigtiff=False )

    def compute_pixel_shift( self ):
        # create a new instance
        rg = registration.ANTSRegisteration()
        if self.ants_path is not None:
            # set path to ANTS executables
            rg.set_ants_path( self.ants_path )
        # verbose settings
        rg.set_verbose( self.verbose )
        # set parameters
        rg.set_ants_params( self.registration_params )

        for (k, cr) in enumerate( self.crop_ranges ):
            fixed_img = self.outdir + 'fixed_{:02d}.tiff'.format(k)
            moving_img = self.outdir + 'moving_{:02d}.tiff'.format(k)
            outdir = self.outdir + '{:02d}'.format(k) + '/'

            rg.set_fixed_im( fixed_img, 1 )
            rg.set_moving_im( moving_img, 1 )
            rg.set_outputs( outdir )

            # run!
            rg.prepare_nifti()
            rg.ANTSRegistration()

    def show_alignment_result( self ):

        for (k, cr) in enumerate( self.crop_ranges ):
            zsl = 10

            # fixed image
            fixpath = self.outdir + '{:02d}'.format(k) + '/fixed_image.nii'
            fix = iut.load_nifti_image( fixpath )
            fix = (self.fixed_scaling * fix).clip( 0, 65535 ).astype( np.uint16 )
            # before warping
            befpath = self.outdir + '{:02d}'.format(k) + '/moving_image.nii'
            bef = iut.load_nifti_image( befpath )
            bef = (self.moving_scaling * bef).clip( 0, 65535 ).astype( np.uint16 )
            # after warping
            aftpath = self.outdir + '{:02d}'.format(k) + '/Warped.nii.gz'
            aft = iut.load_nifti_image( aftpath )
            aft = (self.moving_scaling * aft).clip( 0, 65535 ).astype( np.uint16 )

            # overlaied image, before
            ov1 = np.zeros( (fix.shape[1], fix.shape[2], 3), dtype=np.uint8 )
            # red channel = mCherry = moving
            ov1[:,:,0] = np.clip(bef[zsl,:,:]/255, 0, 255).astype(np.uint8)
            # green channel = fixed
            ov1[:,:,1] = np.clip(fix[zsl,:,:]/255, 0, 255).astype(np.uint8)

            # overlaied image, after
            ov2 = np.zeros( (fix.shape[1], fix.shape[2], 3), dtype=np.uint8 )
            # red channel = moving
            ov2[:,:,0] = np.clip(aft[zsl,:,:]/255, 0, 255).astype(np.uint8)
            # green channel = fixed
            ov2[:,:,1] = np.clip(fix[zsl,:,:]/255, 0, 255).astype(np.uint8)

            fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(5,2.5) )
            ax1.imshow( ov1 )
            ax2.imshow( ov2 )
            ax1.set_title( '(before)' )
            ax2.set_title( '(after)' )

            plt.show()

    def plot_pixel_shit( self ):
    
        result = []
        for (k, cr) in enumerate( self.crop_ranges ):
            matpath = self.outdir + '{:02d}'.format(k) + '/F2M_0GenericAffine.mat'
            mat = sio.loadmat( matpath )
            transl = mat['AffineTransform_double_3_3'][9:12]
            print( transl.ravel() )
            result.append( transl )

        # prepare plot data
        data = np.array( result )
        mean = data.mean( axis=0 )
        sd = data.std( axis=0 )
        labels = [ 'X', 'Y', 'Z' ]
        print( "MEAN:", mean.ravel() )

        # prepare a plot axis
        figsize_mm = (120, 90 )
        figsize_inch  = [ v/25.4 for v in figsize_mm ]
        fig, ax = plt.subplots( figsize=figsize_inch )

        # Horizontal line
        ax.plot([0.8, 1.2], [mean[0], mean[0]], color='teal', linestyle='-', linewidth=2)
        ax.plot([1.8, 2.2], [mean[1], mean[1]], color='teal', linestyle='-', linewidth=2)
        ax.plot([2.8, 3.2], [mean[2], mean[2]], color='teal', linestyle='-', linewidth=2)

        ax.errorbar( [1,2,3], mean, sd, linestyle='None', marker='None',
                     color='teal', linewidth=2, capsize=5)

        for i in range(3):
            y = data[:,i]
            x = np.random.normal(1+i, 0.06, size=len(y))
            ax.plot(x, y, 'o', color='coral', alpha=0.9)

        ax.set_ylabel( 'Shift (pixels)' )
        ax.set_title( 'Channel pixel shift' )
        ax.set_xticks( [1,2,3] )
        ax.set_xticklabels( labels )

        # save plot
        #if not os.path.exists( savedir ):
        #    os.makedirs( savedir )
        #savename = savedir + nickname + '_' + lr + '.png'
        #plt.savefig( savename, dpi=300, transparent=True, bbox_inches='tight' )

        plt.show()


def find_dual_positive( ch1_path, ch2_path, outdir, nickname, dist_thresh, 
                        midline=None,
                        ch2_shift_left=[0,0,0], ch2_shift_right=[0,0,0], verbose=False ):
    """
    find the cell that are positive in both channel 1 and 2.
    INPUTS:
        ch1_path
        ch2_path
        outdir
        nickname
        dist_thresh
        ch2_shift_left
        ch2_shift_right
    """

    # load channel 1
    ch1_full = pd.read_csv( ch1_path )
    # load mCherry cells
    ch2_full = pd.read_csv( ch2_path )
    
    # convert to numpy array
    ch1 = ch1_full[['X', 'Y', 'Z']].values
    ch2 = ch2_full[['X', 'Y', 'Z']].values
    
    # correct pixel shift between two channels; unit is in micron!
    if midline is not None:
        left_idx = ch2[:,0] <= midline
        right_idx = ch2[:,0] > midline
        ch2[left_idx] += np.array( ch2_shift_left )
        ch2[right_idx] += np.array( ch2_shift_right )
    
    # construct KDTree
    if verbose:
        print( "Constructing KDTree..." )
    tree = spatial.cKDTree( ch1 )

    # find nearest neighbor
    if verbose:
        print( "Searching nearest neighbor..." )
    dist, idx = tree.query( ch2, k=1 )
    
    # find ID pairs
    id_pair = []
    ch2_id = 0
    for ch1_id, d in zip(idx, dist):
        if d < dist_thresh:
            id_pair.append( [ch1_id, ch2_id] )
        ch2_id += 1
    id_pair = np.array( id_pair )

    # extract from table
    ch1_table = ch1_full.iloc[ id_pair[:,0] ].copy()
    ch2_table = ch2_full.iloc[ id_pair[:,1] ].copy()

    # get XYZ list
    xyz = ch1_table[['X','Y','Z']].copy()

    # drop XYZ columns
    ch1_table.drop( ['X','Y','Z'], 1, inplace=True )
    ch2_table.drop( ['X','Y','Z'], 1, inplace=True )

    # rename columns, ch1
    ori_name = list( ch1_table )
    name_mapping = {}
    for n in ori_name:
        name_mapping[n] = "ch1_" + n
    ch1_table.rename( index=str, columns=name_mapping,
                      inplace=True )
    
    # rename columns, ch2
    ori_name = list( ch2_table )
    name_mapping = {}
    for n in ori_name:
        name_mapping[n] = "ch2_" + n
    ch2_table.rename( index=str, columns=name_mapping,
                      inplace=True )

    # reset index...
    ch1_table.reset_index( drop=True, inplace=True )
    ch2_table.reset_index( drop=True, inplace=True )
    xyz.reset_index( drop=True, inplace=True )
    # then merge table
    table = pd.concat( [xyz,ch1_table,ch2_table], axis=1 )

    print( "Number of dual positive cells:", table.shape[0] )

    savename = outdir + nickname + '.csv'
    table.to_csv( savename, index=False )




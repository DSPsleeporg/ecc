import h5py
import glob, os, math, warnings, time
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi
import tifffile
from skimage.filters import gaussian
from skimage.transform import downscale_local_mean, resize
from skimage.morphology import erosion

def load_hdf5_image( filepath, dsetname=None, datarange=None ):
	"""
	load HDF5 image
	INPuTS:
	dsetname: dataset name to be loaded.
		       if 'dsetname' is 'None', the first dataset found is used.
	datarange: controls datarange. a list with 6 elements
	e.g.
	vol = load_hdf5_image( 'test.h5', 'data0', [0,100,0,100,0,100] )
	"""
	time.sleep(0.5)
	# sleep is needed because h5py blocks standard output (i.e. 'print') while it's working
	with h5py.File( filepath, 'r', driver='stdio' ) as hf:
		if dsetname is None:
			dsetnames = list( hf.keys() )
			dsetname = dsetnames[0] # use the first dataset
		dset = hf[ dsetname ]

		if datarange is None:
			stack = dset[:]
		else:
			r = datarange
			stack = dset[r[0]:r[1],r[2]:r[3],r[4]:r[5] ]
	# squeeze unncessary dimension
	stack = np.squeeze( stack )

	return stack

def load_tiff_sequence( imgdir, imgtype='tif', range=None ):
	"""
	Load tiff sequence stored in the same directory
	e.g.
	vol = load_tiff_sequence( imgdir, '.png', range=[10,100] )
	"""
	if not imgdir.endswith('/'):
		warnings.warn("'imgdir' must end with '/'", SyntaxWarning)
		imgdir += '/'

	imlist = glob.glob( imgdir + '*.' + imgtype )
	imlist.sort() # sort numerically

	if range is not None:
		imlist = imlist[ range[0]:range[1] ]

	# get image properties by reading the first image
	im = tifffile.imread(imlist[0])
	imsize_x = im.shape[1]
	imsize_y = im.shape[0]
	imsize_z = len( imlist )
	if len( im.shape ) == 3:
		# color image
		color_flag = True
		imsize_c = im.shape[2]
		imsize = ( imsize_z, imsize_y, imsize_x, imsize_c )
	else:
		color_flag = False
		imsize = ( imsize_z, imsize_y, imsize_x )
	imtype = im.dtype

	stack = np.zeros( imsize, dtype=imtype )
	for (i,impath) in enumerate(imlist):
		im = tifffile.imread( impath )
		if color_flag:
			stack[i,:,:, :] = im
		else:
			stack[i,:,:] = im

	return stack

def load_tiff_image( filepath ):
	"""
	load tiffimage (multipage tiff is supported)
	"""
	stack = tifffile.imread( filepath )

	return stack

def load_nifti_image( filepath, dtype='float64' ):
	"""
	load nifti image
	"""
	# this is still in disk
	img = nib.load( filepath )
	# load into memory as ndarray
	stack = img.get_fdata( dtype=dtype )
	#print( "Stack data type:", stack.dtype )
	# swap axis so that 0th index is the z!
	stack = np.swapaxes(stack,0,2)

	return stack

def write_as_hdf5( stack, h5name, dsetname, 
	                chunks_enabled=True, chunksize=None,
	                attributes=None ):
	"""
	e.g.
	write_as_hdf5( vol, 'test.hdf5', 'resolution_0', True, (100,100,100) )
	"""
	if chunks_enabled:
		if chunksize is None:
			chunks = True
		else:
			chunks = chunksize
	else:
		chunks = None

	time.sleep(0.5)
	with h5py.File( h5name, 'w', driver='stdio' ) as hf:
		data = hf.create_dataset( dsetname,
			                       chunks=chunks,
			                       data=stack )
		if attributes is not None:
			for key, value in attributes.items():
				data.attrs[key] = value

def write_as_tiff( stack, tiffname, bigtiff=True ):
	tifffile.imsave( tiffname, stack, bigtiff=bigtiff )

def write_as_nifti( stack, niftiname, spx, spy, spz ):
	"""
	NOTE:
	array layout is reordered so that tiff and nifti can be treated equally;
	In tiff, z is 0th index, i.e. to make xy slice,
	slice = stack[ i, :, : ]
	On the other hand, in nifti, z is the 2nd index, i.e. to make xy slice,
	slice = stack[ :, :, i ]
	"""
	# swap axis!
	stack = np.swapaxes(stack,0,2)
	nim = nib.Nifti1Image( stack, affine=None )
	# define voxel spacing and orientation
	# ANTS uses qform when reading NIFTI-1 images
	aff = np.diag([-spx,-spy,spz,1]) # NOTE! - sign is due to the inconsistency between nibabel and ANTS (or ITK)
	nim.header.set_qform(aff, code=2)
	# write!
	nim.to_filename( niftiname )

def get_hdf5_array_size( filepath, dsetname=None, datarange=None ):
	"""
	"""
	# sleep is needed because h5py blocks standard output (i.e. 'print') while it's working
	with h5py.File( filepath, 'r', driver='stdio' ) as hf:
		if dsetname is None:
			dsetnames = list( hf.keys() )
			dsetname = dsetnames[0] # use the first dataset
		dset = hf[ dsetname ]

		return dset.shape

def downscale_image( stack, scaling ):
	"""
	Downscales an image by a factor defined by 'scaling'
	INPUTS:
	stack: 3D numpy array
	scaling: python list, scaling factor of XYZ
	         e.g. [ 6., 6., 7. ]
	"""
	# define output image size
	imsize_org = np.array( stack.shape )
	imsize_out = imsize_org / np.array( scaling )
	imsize_out = np.round( imsize_out ).astype( np.int )
	imsize_out = imsize_out.tolist()

	# resize image
	resized = resize( stack, imsize_out, preserve_range=True,\
							anti_aliasing=True, mode='reflect' )
	# convert data type
	resized = resized.astype( stack.dtype )
	
	return resized

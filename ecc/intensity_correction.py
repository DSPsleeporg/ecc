import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import h5py, time, os, warnings

from ecc import image_utils as iut

##### PLOT STYLES #####
plt.style.use('seaborn-white')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

def correct_intensity( h5in, h5out, corr_val, dsetname=None ):
	"""
	adjust the intensity values of the input by multiplying with corr_val
	INPUTS:
		h5in:
		h5out:
		corr_val:
	"""
	print("Reading HDF5...")
	time.sleep( 1 )

	with h5py.File( h5in, 'r', driver='stdio' ) as hf:
		if dsetname is None:
			dsetnames = list( hf.keys() )
			dsetname = dsetnames[0] # use the first dataset
		dset = hf[ dsetname ]
		vol = dset[:]
		# get attributes
		attrs = {}
		for key, val in dset.attrs.items():
			attrs[key] = val
		if not attrs: # if attrs is empty
			attrs = None

	print( "Mean before: ", vol.mean() )

	# convert to float32
	vol = vol.astype( np.float32 )
	vol = vol * corr_val
	vol = vol.astype(np.uint16)
	print( "Mean after:", vol.mean() )
	
	outdir = os.path.dirname( h5out )
	if not os.path.exists( outdir ):
			print( "Making a new directory... ", outdir )
			os.mkdir( outdir )

	iut.write_as_hdf5( vol, h5out, dsetname, 
							 chunks_enabled=True,
							 chunksize=(100,100,100),
							 attributes=attrs )

class plot_CDF:
	"""
	plot cumulative distribution function (CDF) of image intensity values
	INPUT:
		hdf5_path:
		threshold: voxels with intensity value smaller than threshold are ignored,
					  and not used when computing CDF
		savedir:
		dsetname: dataset name of HDF5 file to be loaded;
					 by default, the first dataset found gets loaded
		subsampling: subsample array by this factor
						 subsample=2 for example reduces the array size by a factor of 2^3=8
						 this will speed up CDF computation
	"""
	def __init__( self, hdf5_path, threshold, savedir, nickname,
					  dsetname=None, subsampling=1, return_mp=False ):
		self.set_hdf5_path( hdf5_path )
		self.threshold = threshold
		self.set_savedir( savedir )
		self.nickname = nickname
		self.dsetname = dsetname
		self.subsampling = subsampling

		self.run_main()

	def set_hdf5_path( self, hdf5_path ):
		if not os.path.exists( hdf5_path ):
			raise FileNotFoundError("Cannot find ", hdf5_path )
		self.hdf5_path = hdf5_path

		# find base name
		filename = os.path.basename( hdf5_path )
		basename = os.path.splitext( filename )[0]
		self.basename = basename

	def set_savedir( self, savedir ):
		if not savedir.endswith('/'):
			warnings.warn("'savedir' must end with '/'", SyntaxWarning)
			savedir += '/'
		if not os.path.exists( savedir ):
			print( "Making a directory... ", savedir )
			os.mkdir( savedir )
		self.savedir = savedir

	def _write_to_csv( self, csvname, values, counts ):
		d = { 'Intensity_value': values,
		      'Normalized_counts': counts }
		df = pd.DataFrame( data=d )
		df.to_csv( csvname, sep=',', index=False, header=list(df) )

	def _cdf(self, x):
		"""sub-routine;
		computes the CDF of given array input"""
		vals, counts = np.unique(x, return_counts=True)
		ecdf = np.cumsum(counts).astype(np.float64)
		ecdf /= ecdf[-1]
		return vals, ecdf

	def run_main( self ):
		print("Reading HDF5...")
		vol = iut.load_hdf5_image( self.hdf5_path, self.dsetname )
		# sub-sample array
		n = int( self.subsampling )
		vol = vol[::n,::n,::n]
		# make a mask
		mask = vol > self.threshold
		# apply mask
		vol = vol[mask]

		print("Computing CDF...")
		val, cdfval = self._cdf(vol)

		# write CDF values to csv
		csvname = self.savedir + self.nickname + '_CDF_values.csv'
		self._write_to_csv( csvname, val, cdfval )

		# compute middle point where cdfval = 0.5
		idx = np.square( cdfval - 0.5 ).argmin()
		mp = val[ idx ]
		self.mp = mp

		# make a plot!
		f, ax = plt.subplots(1, 1, figsize=(8,5) )
		ax.plot( val, cdfval )
		title = "mean = {:.1f}, middle point = {:.1f}".format( vol.mean(), mp )
		ax.set_title( title )
		ax.set_xlabel("Intensity value")
		ax.set_ylabel("CDF")
		xmin = self.threshold
		xmax = 2 * mp - self.threshold
		ax.set_xlim( [xmin, xmax] )
		ax.set_ylim( [-0.05, 1.05] )
		ax.yaxis.grid(True)		
		# save plot
		plotname = self.savedir + self.nickname + '_CDF.png'
		plt.savefig( plotname, dpi=300, 
						 transparent=True, bbox_inches='tight' )

# class plot_CDF_multi:
# 	def __init__(self, ):

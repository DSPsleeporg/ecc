import os

##################################################################
# Limit the number of threads used by numpy                      #
# This is needed to make it compatible with multiprocessing      #
# see https://github.com/numpy/numpy/issues/11826 for more info! #
##################################################################
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys, time, datetime
import queue as python_queue
import multiprocessing as mlp
import h5py
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import warnings

from ecc import image_utils as iut

class Reader( mlp.Process ):
	"""
	Reader process (subclass of mlp.Process)
	This instance continuously reads small chunk of HDF5 data,
	and pushes it in the queue.
	The data in the queue gets processed by "Worker" instance.
	"""

	def __init__(self, queue, state, verbose=False):
		"""
		INPUTS:
			queue: mlp.Queue object which image blocks are loaded into
			state: mlp.Manager.dictionary object, which stores the state
					 of the parallel processes
		"""
		super(Reader, self).__init__()

		# queue: mlp.Queue object
		self.queue = queue
		# state: mlp.Manager.dictionary
		self.state = state
		self.state["reader_active"] = True
		# verbose level
		self.verbose = verbose

		# some default settings
		self.flag_mask = False

	def set_reader_params(self, blocksize, overlap, buffersize):
		"""
		Set the parameters associated with reader
		INPUT:
			blocksize: size (in voxels) of the image block loaded each time and processed by one worker
			overlap: overlap (in voxels) of the image block
			buffersize: maximum number of image blocks stored in buffer
		Thus, the size of the image loaded each time is blocksize + 2* overlap
		Make buffersize small to save memory usage; the estimated memory consumption is
		(memory size of each image block ) x ( buffersize )
		"""
		self.blocksize = blocksize
		self.overlap = overlap
		self.buffersize = buffersize

	def set_HDF5_paths(self, raw_path, prob_path, mask_path=None):
		"""
		get handles of HDF5 images
		"""
		# get HDF5 handle of raw image
		self.hf_raw, self.dset_raw = self._get_HDF5_handle( raw_path )
		if self.verbose:
			print( "Raw image is ready;", raw_path )
			print( "Data type:", self.dset_raw.dtype )

		# get HDF5 handle of probability image
		self.hf_prob, self.dset_prob = self._get_HDF5_handle( prob_path )
		dt = self.dset_prob.dtype
		if (dt == np.uint8) or (dt == np.uint16):
			self.prob_dtype = dt
		else:
			raise ValueError("probability image must be either uint16 or uint8 datatype!")
		if self.verbose:
			print( "Probaility image is ready;", prob_path )
			print( "Data type:", self.dset_prob.dtype )

		# get HDF5 handle of mask image
		if mask_path is not None:
			self.hf_mask, self.dset_mask = self._get_HDF5_handle( mask_path )
			self.flag_mask = True
			if self.verbose:
				print( "Mask image is ready;", mask_path )
				print( "Data type:", self.dset_mask.dtype )

		# check image size
		if not (self.dset_raw.shape[0:3] == self.dset_prob.shape[0:3]):
			msg = "Critical error! Data shapes are not the same!" +\
					"Raw image shape:" + ','.join( str(x) for x in self.dset_raw.shape ) +\
					"Prob image shape:" + ','.join( str(x) for x in self.dset_prob.shape )
			raise ValueError( msg )

		# get image size
		self.imsizex = self.dset_raw.shape[2]
		self.imsizey = self.dset_raw.shape[1]
		self.imsizez = self.dset_raw.shape[0]
		if self.verbose:
			print( "Image shape:", self.dset_raw.shape )

	def _get_HDF5_handle( self, h5path, dsetname=None ):
		"""Helper function"""
		hf = h5py.File( h5path, 'r' )
		if dsetname is None:
			dsetnames = list( hf.keys() )
			dsetname = dsetnames[0] # use the first dataset
		dset = hf[ dsetname ]

		return hf, dset

	def release_handles( self ):
		"""Release handles of HDF5 iamges"""
		self.hf_raw.close()
		self.hf_prob.close()
		if self.flag_mask:
			self.hf_mask.close()

	def define_blocks( self ):
		"""
		compute the slice indices, to divide the input image into small blocks
		self.blocklist is the list of the slice indices
		"""
		blksize = self.blocksize
		ov = self.overlap
		# X ranges
		xs = list( range(0, self.imsizex, blksize) ) # start index
		xe = [val+blksize for val in xs ] # last index
		if xe[-1] > self.imsizex:
		   xe[-1] = self.imsizex
		# overlap
		xs = [ val-ov if val-ov>=0 else 0 for val in xs ]
		xe = [ val+ov if val+ov<=self.imsizex else self.imsizex for val in xe ]

		# Y ranges
		ys = list( range(0, self.imsizey, blksize) ) # start index
		ye = [val+blksize for val in ys ] # last index
		if ye[-1] > self.imsizey:
		   ye[-1] = self.imsizey
		# overlap
		ys = [ val-ov if val-ov>=0 else 0 for val in ys ]
		ye = [ val+ov if val+ov<=self.imsizey else self.imsizey for val in ye ]

		# Z ranges
		zs = list( range(0, self.imsizez, blksize) ) # start index
		ze = [val+blksize for val in zs ] # last index
		if ze[-1] > self.imsizez:
		   ze[-1] = self.imsizez
		# overlap
		zs = [ val-ov if val-ov>=0 else 0 for val in zs ]
		ze = [ val+ov if val+ov<=self.imsizez else self.imsizez for val in ze ]

		blocklist = []
		for z1, z2 in zip(zs,ze):
		   for y1, y2 in zip(ys,ye):
		       for x1,x2 in zip(xs,xe):
		           blocklist.append([z1,z2,y1,y2,x1,x2])

		self.blocklist = blocklist

	def load_ith_block( self, blockidx ):
		"""
		load the ith image block
		"""
		s = self.blocklist[blockidx]
		# load raw image
		raw = self.dset_raw[ s[0]:s[1], s[2]:s[3], s[4]:s[5] ]

		# load probability iamge
		prob = self.dset_prob[ s[0]:s[1], s[2]:s[3], s[4]:s[5] ]
		# squeeze unncessary dimension
		prob = np.squeeze( prob )
		# normalize probability image in range [0,255]
		if self.prob_dtype == np.uint8:
			pass
		elif self.prob_dtype == np.uint16:
			prob = prob / 255
			prob = prob.clip(0,255).astype(np.uint8)

		# load mask image
		if self.flag_mask:
			mask = self.dset_mask[ s[0]:s[1], s[2]:s[3], s[4]:s[5] ]
			mask = mask > 0
		else:
			mask = np.zeros( raw.shape, dtype="bool" )

		return raw, prob, mask

	def run( self ):
		"""main"""
		try:
			idx = 0
			while idx < len(self.blocklist):
				# check buffered data size
				buffered = self.queue.qsize()
				self.state["num_elements_in_rqueue"] = buffered
				if buffered > self.buffersize:
					time.sleep( 0.25 ) #if queue is full, wait
				else:
					raw, prob, mask = self.load_ith_block( idx )
					block = self.blocklist[idx]
					self.queue.put( [raw, prob, mask, block] ) # enqueue new item
					idx += 1
		finally:
			# when all data have been read, change reader state
			self.state["reader_active"] = False
			# release handle of HDF5 files
			self.release_handles()

class Worker( mlp.Process ):
	"""
	Worker process:
	This instance accepts image data from Reader process through queue,
	and find individual objects (cells).
	The output is sent to Writer process, where results gets written to a file
	"""

	def __init__( self, my_id, queue_reader, queue_writer, 
					  state, verbose=False ):
		"""
		INPUTS:
			my_id: ID (integer value) unique to this worker
			queue_reader: mlp.Queue object where image blocks are buffered
			queue_writer: mlp.Queue object where cell counting result gets written
			state: mlp.Manager.dictionary object, which is used to exchange the state of the processes 
		"""
		super(Worker, self).__init__()

		# ID of this worker
		self.my_id = my_id

		# reader and writer queue
		self.rqueue = queue_reader
		self.wqueue = queue_writer
		
		self.state = state

		# verbose level
		self.verbose = verbose

	def set_worker_params( self, params ):
		"""
		input should be a dictionary, which should contain:
			prob_threshold: probability threshold in range [0, 1.0]
			min_volume: minimum volume of the object.
							if an object is smaller than this, it is neglected
			max_volume: maximum volume of the object.
							if an object is larger than this, it gets divided into several objects
			intensity_mode: how intensity of each object is measured. available options are
								 "max", "local_mean", "obj_mean"
			local_max_rad:
			local_min_rad: the radius of the minimum filter,
								used to compute the intensity background level
			local_min_percentile: percentile value which determines the intensity background
		"""
		self.prob_threshold = params["prob_threshold"]
		self.min_volume = params["min_volume"]
		self.max_volume = params["max_volume"]
		self.intensity_mode = params["intensity_mode"]
		self.local_max_rad = params["local_max_rad"]
		self.local_min_rad = params["local_min_rad"]
		self.local_min_percentile = params["local_min_percentile"]

		self.nbr_size = int( np.power( self.max_volume, 1.0/3 ).round() )

	def set_reader_params( self, blocksize, overlap ):
		"""
		worker has to know the parameters of Reader process, so that it can properly crop the overlaps
		"""
		self.blocksize = blocksize
		self.overlap = overlap

	def count_cells( self, rawim, probim, maskim ):

		raw = rawim.copy()
		# binarize probability
		# THIS ASSUMES THAT probim HAS VALUE RANGE OF [0,255]!
		# This normalization is handled by Reader process
		binary = (probim > self.prob_threshold * 255)
		# apply mask
		binary[ maskim ] = 0

		# define connectivity
		struct = ndi.generate_binary_structure(3,1)
		# label objects
		lbl, _ = ndi.label( binary, struct )
		# unique ids and counts
		unique, counts = np.unique( lbl, return_counts=True )

		# if no object was found, continue
		if unique.max() == 0:
			return []

		# filter objects by its volume
		small, medium, large = [], [], []
		for uq, ct in zip( unique, counts ):
			if uq == 0:
				continue # skip zero!
			if ct <= self.min_volume:
				small.append( [uq, ct] ) # if object is smaller than mimimum size, it gets excluded
			elif self.min_volume < ct <= self.max_volume:
				medium.append( [uq, ct] )
			else:
				large.append( [uq, ct] )

		# list to store detected objects
		detected_obj = []

		# take care of medium objects
		obj_ids = [ e[0] for e in medium ]
		volumes = [ e[1] for e in medium ]
		if obj_ids: # skip if empty
			coms = ndi.center_of_mass( binary, lbl, obj_ids )
			coms = np.array( coms ).round().astype( np.int ) # convert to integer
			for i, com in enumerate( coms ):
				this_idx = obj_ids[i]
				deltaI, bg = self._get_intensity( raw, lbl, this_idx, com,
															 self.intensity_mode,
															 self.local_max_rad, self.local_min_rad,
															 self.local_min_percentile )
				vol = volumes[i]
				obj = [ com[2], com[1], com[0], deltaI, bg, vol ] # X, Y, Z, intensity, volume
				detected_obj.append( obj )

		# take care of large objects
		obj_ids = [ e[0] for e in large ]
		if obj_ids: # skip if empty
			coms = self._com_large_obj( raw, lbl, obj_ids ) # centers of mass and its ID
			for com in coms:
				this_idx = com[3]
				xyz = com[0:3]
				deltaI, bg = self._get_intensity( raw, lbl, this_idx, xyz,
															 mode="local_mean",
															 max_rad=2, min_rad=self.local_min_rad,
															 min_percentile=self.local_min_percentile )
				vol = self.max_volume
				obj = [ com[2], com[1], com[0], deltaI, bg, vol ] # X, Y, Z, intensity, volume
				detected_obj.append( obj )

		return detected_obj

	def _com_large_obj( self, rawimg, label, obj_ids ):
		"""
		Return:
		   centers: list of tuples of length 4
		"""
		# make a mask image by label image
		mask = np.isin( label, obj_ids )
		img = rawimg.copy()

		# add random values to prevent multiple local peaks
		if img.max() > 1000:
			img = img - np.random.randint(1,100,size=img.shape)
			img = img.clip(0,65535).astype(np.uint16)

		# maximum filter
		img_max = ndi.filters.maximum_filter( img, self.nbr_size )
		# find local maximum
		maxima = (img == img_max)
		# apply mask
		maxima[ np.logical_not(mask) ] = 0

		# get array indices
		arr_idx = np.where( maxima > 0 )
		centers = []
		for i in range( arr_idx[0].size ):
			c0 = arr_idx[0][i]
			c1 = arr_idx[1][i]
			c2 = arr_idx[2][i]
			ID = label[c0, c1, c2]
			centers.append( (c0,c1,c2,ID) )

		return centers

	def _get_intensity( self, blk, label, obj_id, arr_idx,
	                    mode='max', max_rad=1, min_rad=3, min_percentile=5 ):
		"""
		"""
		block = blk.copy()
		c0, c1, c2 = arr_idx

		# compute bagkground level
		local = self._get_local_voxels( block, c0, c1, c2, min_rad )
		bg = np.percentile( local, min_percentile )

		if mode=='max':
			local = self._get_local_voxels( block, c0, c1, c2, max_rad )
			sig = local.max()
		elif mode=='local_mean':
			local = self._get_local_voxels( block, c0, c1, c2, max_rad )
			sig = local.mean()
		elif mode=='obj_mean':
			sig = ndi.mean( block, label, obj_id )

		#compute delta
		delta = sig - bg
		if delta < 0:
			delta = 0

		return delta, bg

	def _get_local_voxels( self, block, zc, yc, xc, rad ):
		"""
		"""
		xst = xc-rad if xc-rad>=0 else 0
		yst = yc-rad if yc-rad>=0 else 0
		zst = zc-rad if zc-rad>=0 else 0
		xen = xc+rad+1 if xc+rad+1<=block.shape[2] else block.shape[2]
		yen = yc+rad+1 if yc+rad+1<=block.shape[1] else block.shape[1]
		zen = zc+rad+1 if zc+rad+1<=block.shape[0] else block.shape[0]

		return block[ zst:zen, yst:yen, xst:xen ]

	def _remove_overlaping_points(self, points, block ):
		bs = self.blocksize 
		ov = self.overlap

		# Z clip range
		z_clip_s = ov
		z_clip_e = bs+ov-1
		if block[0]==0:
			z_clip_s = 0
			z_clip_e = bs -1
		if block[1]==self.imsizez:
			z_clip_e = bs+2*ov

		# Y clip range
		y_clip_s = ov
		y_clip_e = bs+ov-1
		if block[2]==0:
			y_clip_s = 0
			y_clip_e = bs -1
		if block[3]==self.imsizey:
			y_clip_e = bs+2*ov

		# X clip range
		x_clip_s = ov
		x_clip_e = bs+ov-1
		if block[4]==0:
			x_clip_s = 0
			x_clip_e = bs -1
		if block[5]==self.imsizez:
			x_clip_e = bs+2*ov

		#z_clip_s = 0 if block[0]==0 else ov
		#z_clip_e = bs+2*ov if block[1]==self.imsizez else bs+ov-1
		# Y clip range
		#y_clip_s = 0 if block[2]==0 else ov
		#y_clip_e = bs+2*ov if block[3]==self.imsizey else bs+ov-1
		# X clip range
		#x_clip_s = 0 if block[4]==0 else ov
		#x_clip_e = bs+2*ov if block[5]==self.imsizex else bs+ov-1

		cleaned = []
		for p in points:
			if (p[0] < x_clip_s) or (p[1] < y_clip_s) or (p[2] < z_clip_s):
				continue 
			if (p[0] > x_clip_e) or (p[1] > y_clip_e) or (p[2] > z_clip_e):
				continue
			cleaned.append( p )

		return cleaned

	def _add_block_offset(self, points, block ):
		xoffset = block[4]
		yoffset = block[2]
		zoffset = block[0]

		offseted = []
		for p in points:
			p[0] += xoffset
			p[1] += yoffset
			p[2] += zoffset
			offseted.append( p )

		return offseted

	def run( self ):

		while True:
			try:
				raw, prob, mask, block = self.rqueue.get( timeout=1 )
			except python_queue.Empty:
				if not self.state["reader_active"]:
					break # if queue is empty and reader_state is inactive, exit!
				else:
					time.sleep(0.2)
					continue # otherwise, wait for new item

			points = self.count_cells( raw, prob, mask )
			if not points:
				self.wqueue.put( [] )
			else:
				cleaned_points = self._remove_overlaping_points( points, block )
				cleaned_points = self._add_block_offset( cleaned_points, block )
				self.wqueue.put( cleaned_points )
			self.state["working_block"] = block

		# if everything is done, change worker state
		self.state["worker_active"][ self.my_id ] = False

class Writer( mlp.Process ):
	def __init__(self, queue_writer, state, verbose=False ):
		"""
		INPUTS:
			queue_writer: mlp.Queue object where cell counting result gets written
			state: mlp.Manager.dictionary object, which is used to exchange the state of the processes 
		"""
		super(Writer, self).__init__()

		self.wqueue = queue_writer
		self.state = state
		self.verbose = verbose

		# some default settings
		self.print_first_line = True

	def set_writer_params( self, params ):

		self.outdir = params["outdir"]
		self.nickname = params["nickname"]
		self.vx_size_x = params["vx_size_x"]
		self.vx_size_y = params["vx_size_y"]
		self.vx_size_z = params["vx_size_z"]
		self.blocklist = params["blocklist"]

	def print_progress( self, percent, num_cells ):

		if self.print_first_line:
			# print the column headers
			h1 = "progress".center(10)
			h2 = "num_cells".center(12)
			h3 = "working_block".center(32)
			h4 = "elements_in_rqueue".center(20)
			header = '|'.join( [h1, h2, h3, h4] )
			print( header )
			self.print_first_line = False

		sys.stdout.write('\r')
		msg1 = "{:.2f}%".format(percent).center(10)
		msg2 = "{:d}".format(num_cells).center(12)
		msg3 = ','.join(str(x) for x in self.state["working_block"]).center(32)
		msg4 = "{:d}".format( self.state["num_elements_in_rqueue"] ).center(20)
		msg = '|'.join( [msg1, msg2, msg3, msg4] )
		sys.stdout.write( msg )
		sys.stdout.flush()

	def _make_pandas_table( self, cells ):
		"""
		convert a list of tuples into a panda table
		Return:
		   table: the columns are [ 'X', 'Y', 'Z', 'deltaI', 'BG', 'volume' ]
		   X, Y, Z are in um!
		"""
		table = pd.DataFrame( data=cells,
		                      columns=[ 'X', 'Y', 'Z', 'deltaI', 'BG', 'vol' ] )
		
		table['deltaI_total'] = table['deltaI'].values * table['vol'].values

		# convert from voxel to um
		table['X'] *= self.vx_size_x
		table['Y'] *= self.vx_size_y
		table['Z'] *= self.vx_size_z
		table['vol'] *= (self.vx_size_x * self.vx_size_y * self.vx_size_z)

		return table

	def run( self ):

		detected_cells = []
		progress = 0
		while True:
			self.print_progress( 100 * progress / len(self.blocklist), len(detected_cells) )
			try:
				cells = self.wqueue.get( timeout=1 )
			except python_queue.Empty:
				#keys = [ "worker_active_{:d}".format( i ) for i in rang]
				#worker_states = self.state["worker_active"]
				if np.any( self.state["worker_active"] ):
					time.sleep(0.2)
					continue # wait for next queue element
				else:
					break # if queue is empty and reader_state is inactive, exit!
	
			if cells: # if cells is empty...
				if not detected_cells:
					detected_cells = cells
				else:
					detected_cells += cells
			progress += 1

		self.print_progress( 100 * progress / len(self.blocklist), len(detected_cells) )
		table = self._make_pandas_table( detected_cells )
		
		print( "\nCell counting successfully completed!" )
		print( "Total number of cells:", table.shape[0] )

		# save results
		savename = self.outdir + self.nickname + '_cells.csv'
		table.to_csv( savename,
						  sep=',', index=False,
						  float_format='%.2f' )
		print( "Cell table was saved as", savename )
		self.state["table_path"] = savename

class CellFinder:

	def __init__( self ):

		# default parameters
		self.verbose = True
		self.num_workers = mlp.cpu_count()
		self.nickname = "untitled"
		self.blocksize = 120
		self.overlap = 20
		self.buffersize = 200
		self.min_volume = 0
		self.max_volume = 1e12
		#self.nbr_size = 4
		self.prob_threshold = 0.7
		self.intensity_mode = 'max'
		self.local_max_rad = 1
		self.local_min_rad = 4
		self.local_min_percentile = 5

	def set_num_workers( self, num_workers ):
		if num_workers < 1:
			raise ValueError( "'num_workers' must be >= 1!" )
		self.num_workers = num_workers

	def set_verbose( self, TorF ):
		"""if True, print detailed outputs"""
		self.verbose = TorF

	def set_nickname( self, nickname ):
		"""give it a nice nickname"""
		self.nickname = nickname

	def set_outdir( self, outdir ):
		"""set directory to save results"""
		if not outdir.endswith('/'):
			warnings.warn("'outdir' must end with '/'!", SyntaxWarning)
			outdir += '/'
		if not os.path.exists(outdir):
			print("'outdir' does not exit. Creating a new directory...")
			os.mkdir(outdir)
		self.outdir = outdir

	def set_image_voxel_size( self, vx_size ):
		"""
		set voxel size of the input image in micron,
		'vx_size' should be a dictionary of three elements,
		e.g. vx_size = {"X": 6.45, "Y":6.45, "Z":6.45 }
		"""
		self.vx_size_x = vx_size["X"]
		self.vx_size_y = vx_size["Y"]
		self.vx_size_z = vx_size["Z"]

	def set_block_size( self, blksize ):
		if blksize["blocksize"] < 50:
			raise ValueError( "'blocksize' must be larger than 50" )
		self.blocksize = blksize["blocksize"]
		if blksize["overlap"] < 0:
			raise ValueError( "'overlap' must be > 0")
		self.overlap = blksize["overlap"]

	def set_buffersize( self, buffersize ):
		self.buffersize = buffersize

	def set_min_particle_volume( self, min_volume ):
		"""set minimum volume of the particle (in voxels)"""
		if min_volume < 0:
			raise ValueError( "'min_volume' must be > 0" )
		self.min_volume = min_volume

	def set_max_particle_volume( self, max_volume ):
		"""
		set maximum volume of the particle (in voxels)
		paritcles larger than this volume gets passed to large object routine,
		to separate it into multiple particles
		"""
		if max_volume < 0:
			raise ValueError( "'max_volume' must be > 0" )
		self.max_volume = max_volume

	# def set_neighborhood_size( self, neighborhood_size ):
	# 	"""set size of the maximum filter (in voxels)"""
	# 	if neighborhood_size < 0:
	# 		raise ValueError( "'neighborhood_size' must be > 0" )
	# 	self.nbr_size = neighborhood_size

	def set_prob_threshold( self, prob_threshold ):
		"""set threshold value for probability image.
		should be in range [0,1.0]"""
		if (prob_threshold < 0) or (prob_threshold > 1):
			raise ValueError("'prob_threshold should be in range [0,1.0]")
		self.prob_threshold = prob_threshold

	def set_intensity_computation_mode( self, mode ):
		"""set how intensity value of detected cells are compuated"""
		if not mode in [ "max", "local_mean", "obj_mean" ]:
			raise ValueError( "Invalid intensity mode!", mode )
		self.intensity_mode = mode
	
	def set_local_max_rad( self, local_max_rad ):
		"""set the radius to compute maximum intensity"""
		if local_max_rad < 0:
			raise ValueError( "'local_max_rad' must be > 0" )
		self.local_max_rad = local_max_rad

	def set_local_min_rad( self, local_min_rad ):
		"""set the radius to compute the background lavel"""
		if local_min_rad < 0:
			raise ValueError( "'local_min_rad' must be > 0" )
		self.local_min_rad = local_min_rad

	def set_local_min_percentile( self, local_min_percentile ):
		""" """
		if (local_min_percentile < 0) or (local_min_percentile > 100):
			raise ValueError( "'local_min_percentile' must be in range [0, 100]" )
		self.local_min_percentile = local_min_percentile

	def set_prob_image_path( self, filepath ):
		if not os.path.exists( filepath ):
			raise ValueError( "Cannot find probability image!", filepath )
		self.prob_im_path = filepath

	def set_raw_image_path( self, filepath ):
		if not os.path.exists( filepath ):
			raise ValueError( "Cannot find raw image!", filepath )
		self.raw_im_path = filepath

	def set_mask_image_path( self, filepath ):
		if not os.path.exists( filepath ):
			raise ValueError( "Cannot find mask image!", filepath )
		self.mask_im_path = filepath

	# def plot_intensity_profile( self, use_column='deltaI', intensity_range=None ):
	# 	"""
	# 	using the detected cell table, plot the intensity profile
	# 	"""
	# 	if self.verbose:
	# 		print( "Making a intensity profile plot..." )
	# 	if not hasattr(self, 'table_path'):
	# 		raise ValueError("cell table is not ready")

	# 	savename = self.outdir + 'intensity_profile.png'
	# 	plot_funcs.plot_intensity_profile( self.table_path, self.nickname,
	# 	                                   use_column, savename, intensity_range )
	
	# def plot_SBR( self, plot_x_range=None ):
	# 	"""
	# 	using the detected cell table, plot the SBR profile
	# 	"""
	# 	if self.verbose:
	# 		print( "Making a signal-to-background ratio (SBR) plot..." )
	# 	if not hasattr(self, 'table_path'):
	# 		raise ValueError("cell table is not ready")

	# 	savename = self.outdir + 'SBR_profile.png'
	# 	plot_funcs.plot_SBR( self.table_path, self.nickname,
	# 	                     savename, plot_x_range )

	# def plot_cell_positions( self ):
	# 	"""
	# 	plot detected cell positions, and saves it as a image (NIFTI format)
	# 	"""
	# 	if self.verbose:
	# 		print( "Plotting detected cell positions..." )
	# 	if not hasattr(self, 'table_path'):
	# 		raise ValueError("cell table is not ready")

	# 	savename = self.outdir + 'cell_positions.nii.gz'
	# 	imgshape = (self.imsizez, self.imsizey, self.imsizex)
	# 	vx_size = {"X": self.vx_size_x, "Y": self.vx_size_y,
	# 	          "Z": self.vx_size_z }

	# 	plot_funcs.plot_cell_positions( self.table_path, savename,
	# 	                               imgshape, vx_size )

	def run_main( self ):

		ctime = datetime.datetime.now().strftime('%H:%M:%S')
		print( ctime, ": Starting cell counting." )

		# define shared object to exchange process states
		mgr = mlp.Manager()
		state = mgr.dict()
		state["num_elements_in_rqueue"] = 0
		state["working_block"] = [0,0,0,0,0,0]
		state["reader_active"] = True
		state["worker_active"] = mgr.list( [True for i in range(self.num_workers)] ) # shared objects can be nested!

		# define queue to put data
		rque = mlp.Queue()
		wque = mlp.Queue()

		# initialize reader process
		reader = Reader( rque, state, verbose=True )
		reader.set_reader_params( self.blocksize, self.overlap, self.buffersize )
		if hasattr( self, "mask_im_path" ):
			reader.set_HDF5_paths( self.raw_im_path, self.prob_im_path, self.mask_im_path )
		else:
			reader.set_HDF5_paths( self.raw_im_path, self.prob_im_path )
		reader.define_blocks()
		# store some properties
		self.blocklist = reader.blocklist
		self.imsizez = reader.imsizez
		self.imsizey = reader.imsizey
		self.imsizex = reader.imsizex

		# initialize worker process 
		workers = []
		for i in range( self.num_workers ):
			w = Worker( i, queue_reader=rque, queue_writer=wque, state=state )
			w.set_worker_params({
				"prob_threshold": self.prob_threshold,
				"min_volume": self.min_volume,
				"max_volume": self.max_volume,
				"intensity_mode": self.intensity_mode,
				"local_max_rad": self.local_max_rad,
				"local_min_rad": self.local_min_rad,
				"local_min_percentile": self.local_min_percentile
			})
			w.set_reader_params( self.blocksize, self.overlap )
			w.imsizez = self.imsizez
			w.imsizey = self.imsizey
			w.imsizex = self.imsizex
			workers.append( w )

		# initialize writer process
		writer = Writer( wque, state, verbose=True )
		writer.set_writer_params({
			"outdir": self.outdir,
			"nickname": self.nickname,
			"vx_size_x": self.vx_size_x,
			"vx_size_y": self.vx_size_y,
			"vx_size_z": self.vx_size_z,
			"blocklist": self.blocklist
		})

		start_time = time.time()
		# start process!!
		reader.start()
		time.sleep( 3 )
		for w in workers:
			w.start()
		time.sleep( 1 )
		writer.start()

		# join processes
		writer.join()
		for w in workers:
			w.join()
		reader.join()
		
		elapsed_time = time.time() - start_time
		ctime = datetime.datetime.now().strftime('%H:%M:%S')
		print( ctime, ": Finished cell counting." )
		print( "Elapsed time: {:.1f} s".format(elapsed_time) )

		self.table_path = state["table_path"]
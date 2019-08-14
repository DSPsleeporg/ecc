import subprocess
import multiprocessing
import os, sys, time, datetime

class PixelClassifier:
	def __init__(self):
		self._set_default_settings()

	def set_project_file(self, project_path):
		"""set the ilastik project file
		   project_file should have .ilp extension"""
		if not os.path.exists(project_path):
			raise FileNotFoundError('cannot find', project_path)
		if project_path.endswith('.ilp'):
			self.ilp = project_path
		else:
			raise ValueError("ilastik project file should have .ilp extension!")

	def set_input_image(self, image_path):
		"""set the path to input image
			input image should have .h5 or hdf5 extension"""
		if not os.path.exists(image_path):
			raise FileNotFoundError( f'cannot find {image_path}' )
		if image_path.endswith(('.h5', 'hdf5')):
			self.imgpath = image_path
		else:
			raise ValueError("input image must have .hdf5 or .h5 extension!")

	def set_ilastik_executable_path(self, path):
		"""set the path to the ilastik executables"""
		il_path = os.path.join(path, 'run_ilastik.sh')
		if os.path.exists(il_path):
			self.il_path = il_path
		else:
			msg = "Cannot find 'run_ilastik.sh'. Check your ilastik path again!"
			raise ValueError(msg)

	def set_num_threads(self, num_threads):
		"""set number of threads allocated for ilastik classifier"""
		self.num_threads = int(num_threads)

	def set_max_memory_size(self, max_ram):
		"""set the maximum memory amount (in MB) allocated for ilastik classifier"""
		self.max_ram = int(max_ram)

	def set_output_dir(self, outdir):
		"""set directory to save results"""
		if not os.path.exists(outdir):
			if self.verbose:
				print(f"{outdir} does not exit. Creating a new directory...")
			os.mkdir(outdir)
		self.outdir = outdir

	def set_basename( self, basename ):
		"""set the basename of the output file"""
		self.basename = basename

	def set_verbose(self, TorF ):
		"""if verbose is True, more information is printed on the standard output"""
		self.verbose = TorF

	def _set_default_settings(self):
		"""set the default settings"""
		self.num_threads = multiprocessing.cpu_count() # use all CPU cores
		self.max_ram = 5000 # 5,000MB = 5GB
		self.basename = "prob_image"
		self.verbose = True

	# def runShellCommand(self, cmd, verbose=True):
	# 	"""
	# 	Runs a command on the shell. The output is printed 
	# 	as soon as stdout buffer is flushed
	# 	"""
	# 	pr = subprocess.Popen( cmd, shell=True, stdout=subprocess.PIPE,
	# 	                       stderr=subprocess.STDOUT )
	# 	while pr.poll() is None:
	# 		line = pr.stdout.readline()
	# 		if line != '':
	# 			if verbose: print( line.decode('utf-8').rstrip() )
		
	# 	# Sometimes the process exits before we have all of the output, so
	# 	# we need to gather the remainder of the output.
	# 	remainder = pr.communicate()[0]
	# 	if remainder:
	# 		if verbose: print( remainder.decode('utf-8').rstrip() )

	# 	rc = pr.poll()
	# 	return rc

	def _runShellCommand2(self, cmd, verbose=True):
		"""
		Runs a command on the shell. The output is printed 
		as soon as stdout buffer is flushed
		"""
		pr = subprocess.Popen( cmd, shell=True, stdout=subprocess.PIPE,
		                       stderr=subprocess.STDOUT )
		iters = 0
		while pr.poll() is None:
			line = pr.stdout.readline()
			if line != '':
				if verbose:
					msg = line.decode('utf-8').rstrip()
					if msg.startswith('INFO'):
						if iters > 0: print("\n")
						print( msg )
					elif msg.startswith('DEBUG'):
						log = f"Iteration {iters:05d}: "
						iters += 1
						msg = msg[46:] # skip the first characters
						sys.stdout.write('\r')
						sys.stdout.write(log + msg + ' '*20)
						sys.stdout.flush()
		
		# Sometimes the process exits before we have all of the output, so
		# we need to gather the remainder of the output.
		remainder = pr.communicate()[0]
		if remainder:
			if verbose: print( remainder.decode('utf-8').rstrip() )

		rc = pr.poll()
		return rc

	def _compile_command(self):
		"""
		Return:
			compiled command
		"""
		ilmain = self.il_path + " --headless"
		thrd = "LAZYFLOW_THREADS=" + str( self.num_threads )
		mem = "LAZYFLOW_TOTAL_RAM_MB=" + str( self.max_ram )
		prj = "--project=" + self.ilp
		outp = "--output_filename_format=" + self.outdir + self.basename
		inpt = self.imgpath

		opt = [ "--export_source='probabilities'",
				"--output_format=hdf5",
				"--output_internal_path=probability",
				"--cutout_subregion='[(None,None,None,0), (None,None,None,1)]'",
				"--export_dtype=uint8",
				"--pipeline_result_drange='(0.0,1.0)'",
				"--export_drange='(0,255)'" ]
		opt = " \\\n".join( opt )

		cmd = " \\\n".join( [thrd, mem, ilmain, prj, outp, opt, inpt] )

		return cmd

	def run(self, just_checking_command=False):
		"""
		run ilastik classifier
		"""
		cmd = self._compile_command()

		if just_checking_command:
			print( cmd )
			return
		
		print("Running ilastik pixel classifier...")
		print("#"*30)
		print(cmd)
		print("#"*30)
		print()

		# run!
		start_time = time.time()
		rc = self._runShellCommand2(cmd, self.verbose)
		elapsed_time = time.time() - start_time

		# Check if the computation was successful
		if rc==0:
			print()
			print("ilastik classifier successfully completed!")
			print(f"Probability image saved in {self.outdir}")
			print(f"Elapsed time: {elapsed_time:.2f} s")
		else:
			print()
			msg = f"ilastik classifier failed! Return code: {rc}"
			raise RuntimeError(msg)
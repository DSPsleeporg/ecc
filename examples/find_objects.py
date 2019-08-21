from ecc.cellfinder import CellFinder

# create a new instance
cf = CellFinder()

# force it print detailed progress
cf.set_verbose(True)

# Optional: set number of parallel CPU cores
cf.set_num_workers(4)

# set voxel size of the input image
cf.set_image_voxel_size( {"X": 6.45, "Y": 6.45, "Z": 7.0} )

# Optional: set block size
cf.set_block_size({"blocksize": 120, "overlap": 20})

# set some parameters
cf.set_prob_threshold(0.70)
cf.set_min_particle_volume(2)
cf.set_max_particle_volume(64)
cf.set_intensity_computation_mode('obj_mean')

# give it a nice nickname
cf.set_nickname('test')
# set output directory
cf.set_outdir('/data/cellfinder-result/')

cf.set_raw_image_path('/data/test-image.hdf5')
cf.set_prob_image_path('/data/classifier-result/prob_image.h5')

# run!
cf.run_main()
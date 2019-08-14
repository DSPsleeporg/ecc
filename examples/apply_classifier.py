from ecc.ilastik_wrapper import PixelClassifier

# create a new instance
pc = PixelClassifier()

# force it print detailed progress
pc.set_verbose(True)

# set ilastik path
pc.set_ilastik_executable_path('/home/users/mano/softwares/ilastik-1.3.0-Linux/')

# set path to ilastik project
pc.set_project_file('/home/users/mano/python-notebook/MATOME/CC-KRV-2/ilastik_training/KRV_classifier_GFP.ilp')

# Optional: control CPU and memory usage
pc.set_num_threads(20)
pc.set_max_memory_size(60000) # in MBs

# define output
pc.set_output_dir('../data/classifier-result/')

# define input image
pc.set_input_image('../data/test-image.hdf5')

# run!
pc.run()
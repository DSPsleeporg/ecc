---
title: Applying ilastik classifier
nav_order: 6
---

# Applying ilastik classifier

Once the ilastik project is created, you can apply the classifier to multiple images that you are analyzing.

To do this, use `PixelClassifier` class from `ecc.ilastik_wrapper` module:

## Example script
<https://github.com/DSPsleeporg/ecc/blob/master/examples/apply_classifier.py>

```python
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
```

To process multiple brains, add for loop in the above code.


## Some more explanation

### Create a new instance
```python
from ecc.ilastik_wrapper import PixelClassifier
pc = PixelClassifier()
```

### (Optional) Set verbose
By telling `set_verbose(True)`, the detailed progress information is printed on the screen output. If you want a silent execution, give it a false value.
```python
pc.set_verbose(True)
```

### Set path to ilastik executable file
Set path to the directory where ilastik executable file is located:
```python
pc.set_ilastik_executable_path('/home/software/ilastik-1.3.0-Linux/')
```

### Set path to ilastik project
Next, tell ecc where your ilastik project file is stored:
```python
pc.set_project_file('../data/classifier.ilp')
```

### (Optional) Control memory and CPU usage
Control the number of threads and maximum memory (in MBs) allocated for ilastik.
```python
pc.set_num_threads(20)
pc.set_max_memory_size(60000)
```

### Set inputs and outputs
Set output directory:
```python
# define output directory
pc.set_output_dir('../data/classifier-result/')
```
By default, probability image will be saved as `{output_dir}/prob_image.h5`.

Set input image:
```python
# define input image
pc.set_input_image('../data/test-image.hdf5')
```

### Run!
Now you are ready to run classifier. Start the computation by
```python
pc.run()
```
Given 15GB image volume, computation takes about 30 to 60 minutes using 30 CPU cores.

If everything is successful, a probability image (in HDF5 format) is produced in the directory you specified. In probability image, the value of each pixel corresponds to the probability of that pixel being label 0 (i.e. cell). Probability values of other labels are discarded (since we do not need them). Probability value is rescaled to unsigned 8 bit integer (i.e. [0, 255]). Using this probability image, we will identify individual cells in the next step.
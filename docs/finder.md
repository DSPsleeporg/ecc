---
title: Segmenting objects
nav_order: 7
---

## Example script

```python
from ecc.cellfinder import CellFinder

# create a new instance
cf = CellFinder()

# force it print detailed progress
cf.set_verbose(True)

# set number of parallel CPU cores
cf.set_num_workers(2)

# set voxel size of the input image
cf.set_image_voxel_size( {"X": 8.25, "Y": 8.25, "Z": 9.0} )

# set some parameters
cf.set_min_particle_volume(2)
cf.set_max_particle_volume(64)
cf.set_prob_threshold(0.70)
cf.set_block_size({"blocksize": 120, "overlap": 20})
cf.set_intensity_computation_mode('obj_mean')

# give it a nice nickname
cf.set_nickname('test')
# set output directory
cf.set_outdir('../data/cellfinder-result/')

cf.set_raw_image_path('../data/test-image.hdf5')
cf.set_prob_image_path('../data/classifier-result/prob_image.h5')

# run!
cf.run_main()
```


## Some more explanation

### 3. Find cells
To extract individual cells from the probability image, use `CellFinder` class from `cloudmap.ecc.cellfinder_parallel`:

```python
from cloudmap.ecc.cellfinder_parallel import CellFinder
cf = CellFinder()
```

#### (Optional) Set the number of threads
Control the number of threads allocated for this computation depending on your computer's availability.
```python
cf.set_num_workers( 35 )
```

#### Set voxel size of the input image
First, set voxel size of the input image **sin &mu;m**:
```python
cf.set_image_voxel_size( {"X": 6.45, "Y": 6.45, "Z": 7.0} )
```
If you want silent execution, give verbose false value.

#### (Optional) Set verbose
If you want silent execution, give verbose false value.
```python
cf.set_verbose( True )
```

#### (Optional) Set blocksizes and overlaps
CellFinder divides the large 3D into small blocks and processes them in a parallel manner. You can adjust the block size and overlap value by
```python
cf.set_block_size( {"blocksize": 150, "overlap": 20} )
```
The default is {"blocksize": 120, "overlap": 20}, and you normally do not need to change this value.

#### Set counting paramters
```python
cf.set_prob_threshold( 0.70 )
cf.set_min_particle_volume( 2 )
cf.set_max_particle_volume( 64 )
cf.set_intensity_computation_mode( 'mean' )
cf.set_block_size( {"blocksize": 150, "overlap": 20} )
```

`prob_threshold` literally means the thresholding value applied to the probability image. Voxels with larger values than threshold are considred as objects (i.e. cell). The value should be in range [0, 1].

`min_particle_volume` defines the minimum volume (in voxel unit, not in &mu;m<sup>3</sup>!) of the segmented objects. Objects smaller than this volume are discarded.

`max_particle_volume` defines the maximum volume (in voxel unit, not in &mu;m<sup>3</sup>!) of the segmented objects. Objects larger than this volume is sent to object separation routine, where the objects gets separeted into multiple objects.

`intensity_computation_mode` defines how the intensity value is computed for each cell. Available options are `max` and `obj_mean`. `max` simply picks up the maximum intensity value of the segmented object. `obj_mean` computes the mean intensity value of the segmented object, calculated as 

$$mean = \frac{\sum Intensity}{ volume}$$

Namely, to recover the total intensity of the object, simply multiply as $$ mean \times volume$$.

#### Set inputs and outpus
```python
cf.set_nickname( "CFM001" )
cf.set_output_dir( './result_CFM001/' )
cf.set_raw_image_path( '../data/counting/CFM001_594_cFos.hdf5' )
cf.set_prob_image_path( '../data/counting/prob_CFM001/594_cFos_probability.h5' )
```

`nickname` is used in file names and plot titles, to keep track of the data.

#### (Optional) Set mask
Optionally, you can define a mask.
```python
cf.set_mask_image_path( '../data/counting/CFM001_594_cFos_mask.hdf5s' )
```
Image mask should be given as HDF5. Voxels with values equal or greater than 1 are regarded as mask area, and objects that are covered by the mask gets discarded.

For example, when strong autofluorescence is present on the brain surface (which is often the case), you can generate a brain surface mask to exclude those signals.

#### Run!
Now everything is ready. Start the main calculation by
```python
cf.run_main()
```
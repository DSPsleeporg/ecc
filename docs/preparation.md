---
title: Data preparation
nav_order: 4
---

# Data preparation
To efficiently process large 3D images (> 10GB), ecc is designed to work with [HDF5](https://www.hdfgroup.org/solutions/hdf5/). In particular, ecc (and ilastik as well) utilizes the HDF5's feature called [chuncked storage](https://support.hdfgroup.org/HDF5/doc/Advanced/Chunking/index.html), which helps to increase the speed to read/write a subvolume of the large 3D array.

To do this image format conversion, you can either write your own code, or use a simple python functions offered by ecc. A bunch of utility functions are defined in `ecc.image_utils` module. The chunk size should usually be about (100,100,100) or similar size.

## Example python script
<https://github.com/DSPsleeporg/ecc/blob/master/examples/convert_to_hdf5.py>

```python
from ecc import image_utils as iut

### Change these variables! ###
f = '../data/test-image.tif'
# Input image file. Supported formats are TIFF (.tif or .tiff), NIfTI (.nii or .nii.gz),
# or TIFF sequence. When loading a TIFF sequence, specify a directory name which should end with /.
out = '../data/tmp.hdf5'
# Output file name. File extension should be either .h5 or .hdf5

# load image
print("Loading input image...")
if f.endswith(('.tif', '.tiff')):
    print("Input image type: 3D TIFF")
    stack = iut.load_tiff_image(f)
elif f.endswith(('.nii', '.nii.gz')):
    print("Input image type: NIfTI")
    stack = iut.load_nifti_image(f)
elif f.endswith('/'):
    print("Input image type: TIFF sequence")
    stack = iut.load_tiff_sequence(f)
else:
    raise ValueError("Invalid input!")

# check output name
if not out.endswith(('.h5', '.hdf5')):
    raise ValueError("Invalid output file extension!")

# write as HDF5. Enable chunked storage
# chunk size is (100,100,100)
print("Writing HDF5...")
iut.write_as_hdf5(
    stack, out, "resolution_0",
    chunks_enabled=True,
    chunksize=(100,100,100)
    )
```


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
    print("Input image type: TIFF")
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

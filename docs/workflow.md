---
title: Workflow overview
nav_order: 2
---

# Workflow overview

## [0. Preparation (Python)](preparation.html)
  * Convert your image data into [HDF5 format](https://www.hdfgroup.org/solutions/hdf5/). To do this, ecc provides a utility function. See [Data preparation](preparation.html) to learn more.
  * If necessary, reorient the image to follow the convention used in CUBIC-Atlas. See [About brain coordinates](coordinate.html) for more details.

## [1. Train ilastik classifier (ilastik GUI)](ilastik.html)

ecc uses [ilastik](https://www.ilastik.org/) to classify voxels into distinct labels (cells, background, etc) powered by machine learning. You must first make a ilastik project (`.ilp` file) and train a classifier for your image set. See [Training ilastik classifier](ilastik.html) to learn about how to train ilastik classifier.

## [2. Apply ilastik classifier (Python)](classifier.html)
Once a good ilastik classifier is trained, you can apply the classifier to a group of images that you are working on. To facilitate this, ecc provides `PixelClassifier` class from `ecc.ilastik_wrapper` module. See [Applying ilastik classifier](classifier.html) for more details.

## [3. Segment individual cells (Python)](finder.html)
ilastik pixel classifier produces a "probability image", where the pixels (voxels) corresponding to desired objects (e.g. cells) are marked with high probability values. From this image, we will segment individual objects and quantify the intensity and the object volume. To do this, use `CellFinder` class from `ecc.cell_finder` module. See [Segmenting cells](finder.html) for more details.
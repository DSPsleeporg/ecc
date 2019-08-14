---
title: Workflow overview
nav_order: 2
---

# Workflow overview

## [0. Preparation (Python)](preparation.html)
  * Convert your image data into [HDF5 format](https://www.hdfgroup.org/solutions/hdf5/). To do this, ecc provides a utility function. See [here](preparation.html) to learn more.
  * If necessary, reorient the image to follow the convention used in CUBIC-Atlas. See [here](coordinate.html) for more details.

## [1. Train ilastik classifier (ilastik GUI)](ilastik.html)

ecc uses [ilastik](https://www.ilastik.org/) to classify voxels into distinct labels (cells, background, etc) by machine learning algorithms. You must first make a ilastik project (`.ilp` file) and train a classifier for your image set. See [here](ilastik.html) to learn about how to train ilastik classifier.

## [2. Apply pixel classifier (Python)](classifier.html)
Once the ilastik project is created, you can apply the classifier to a group of images that you are working on. To do this, ecc provides `PixelClassifier` class from `ecc.ilastik_wrapper` module. See [here](classifier.html) for more details.

## [3. Segment individual cells (Python)](finder.html)
ilastik pixel classifier produces a "probability image", where the pixels (voxels) corresponding to desired objects (e.g. cells) are marked with high probability values. From this image, we will segment individual objects and quantify the intensity as well as the object volume. To do this, use `CellFinder` class from `ecc.cell_finder` module. See [here](finder.html) for more details.
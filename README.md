# easy_cell_counter (ecc)

For more detailed documentation, see https://dspsleeporg.github.io/easy_cell_counter/.

## Installation
### System requirements
ecc has been developed and tested on Linux platform (Ubuntu 16.04 LTS and CentOS 7).

### Install ilastik
ecc uses [ilastik](https://www.ilastik.org/) to train a pixel classifier. Follow [this page](https://www.ilastik.org/documentation/basics/installation) for the installation. For the best compatibility, it is recomennded to use **version 1.3.0 or 1.3.2**.

Be sure to remember where you installed ilastik. We will need that path when we run ecc.

### Install ecc
ecc uses [conda](https://docs.conda.io/projects/conda/en/latest/index.html) to replicate virtual python environment. If you do not have conda installed in your system, you can download miniconda [here](https://docs.conda.io/en/latest/miniconda.html).

First, clone the repo:
```bash
$ git clone https://github.com/DSPsleeporg/easy_cell_counter.git
```

Then, go to the directory and create a new conda environment:

```bash
$ cd easy_cell_counter
$ conda env create -f environment.yml
```

This will create a new conda environment named `ecc` (Note: name conflict may occur if you already have a conda environment named ecc).

Now activate the new environment by:
```bash
$ conda activate ecc
```

Then, **within the virtual environment**, install ecc:
```bash
(ecc) $ pip install .
```



import os
import pandas as pd
import numpy as np

def filter_cells_by_SBR( cell_table_path, sbr_thresh, savename=None ):

    table = pd.read_csv( cell_table_path, header=0 )

    deltaI = table["deltaI"].values
    bg = table["BG"].values
    intensity = deltaI + bg
    # we do not want 0!
    bg[bg==0] = np.nan
    sbr = intensity / bg

    # apply threshold
    result = table.values[ sbr > sbr_thresh ]
    result = pd.DataFrame( columns=list(table), data=result )
    print( "Number of cells before filtering:", table.shape[0] )
    print( "Number of cells after filtering:", result.shape[0] )

    # save as csv
    if savename is None:
        basename = os.path.splitext( cell_table_path )[0]
        savename = basename + '_filtered.csv'
    result.to_csv( savename, index=False )
    print( "Filtered cell table saved as", savename )

def filter_cells_by_intensity( cell_table_path, intensity_thresh, 
                               use_column=None, savename=None ):
    
    table = pd.read_csv( cell_table_path, header=0 )

    if use_column is None:
        use_column = "deltaI"

    deltaI = table["deltaI"].values

    # apply threshold
    result = table.values[ deltaI > intensity_thresh ]
    result = pd.DataFrame( columns=list(table), data=result )
    print( "Number of cells before filtering:", table.shape[0] )
    print( "Number of cells after filtering:", result.shape[0] )

    # save as csv
    if savename is None:
        basename = os.path.splitext( cell_table_path )[0]
        savename = basename + '_filtered.csv'
    result.to_csv( savename, index=False )
    print( "Filtered cell table saved as", savename )


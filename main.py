## Importing packages
import os
import shutil
import glob

## Importing useful scripts
from utils import *



if __name__ == '__main__':

	directory = 'input/309_tm_25C_top_need_35C/'

	crop_bb_all(directory)

	outDir = 'output/309_tm_25C_top_need_35C/'

	generate_mask_all(outDir)
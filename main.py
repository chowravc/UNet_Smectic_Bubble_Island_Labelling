## Importing packages
import os
import shutil
import glob

## Importing useful scripts
from utils import *



if __name__ == '__main__':

	outputDir = 'output/'

	# Create output directory if doesn't exist
	if len(glob.glob(outputDir)) == 0:

		os.mkdir(outputDir)

	# Create mask directory if doesn't exist

	outDir = outputDir + 'example/'

	if len(glob.glob(outDir)) != 0:

		shutil.rmtree(outDir)

	os.mkdir(outDir)

	# Create mask directory

	maskDir = outDir + 'masks/'

	os.mkdir(maskDir)

	imPath = 'input/example/face.jpg'

	generate_mask(imPath)
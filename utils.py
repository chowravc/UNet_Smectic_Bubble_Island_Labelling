## Importing packages
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import shutil

## Outputs smaller of the numbers
def smaller(n1, n2):

	if n1 <= n2:
		return n1
	else:
		return n2


## Getting coefficients of conic section from five points
def fivePointsToConic(points, f = 1.0):
	"""Solve for the coefficients of a conic given five points in Numpy array

	`points` should have at least five rows.

	`f` is the constant that you can specify. With the returned solution,
	`(a, b, c, d, e, f)`, the full conic is specified as:

	$a x^2 + b x y + c y^2 + d x + e y = -f$

	If `points` has exactly five rows, the equation will be exact. If `points`
	has *more* than five rows, the solution will be a least-squares one that
	fits the data the best.
	"""
	from numpy.linalg import lstsq

	x = points[:, 0]
	y = points[:, 1]

	if max(x.shape) < 5:

		raise ValueError('Need >= 5 points to solve for conic section')

	A = np.vstack([x**2, x * y, y**2, x, y]).T
	fullSolution = lstsq(A, f * np.ones(x.size), rcond=None)

	(a, b, c, d, e) = fullSolution[0]

	coeffs = {
		'a':a,
		'b':b,
		'c':c,
		'd':d,
		'e':e,
		'f':f
	}

	return coeffs


## Get five input points from an image
def get_points(im):

	# Clear plots
	plt.clf()

	# Display image
	plt.imshow(im)

	# Get input points
	ps = plt.ginput(5)

	# Close plot
	plt.close()

	# Clear plots
	plt.clf()

	# Return points as list of lists
	return np.array([[p[0], p[1]] for p in ps])


## Generate mask from image
def generate_mask(path, outPath):

	# Clear plots
	plt.clf()

	# Open image
	image = Image.open(path)

	# Store whether mask is correct
	correct = 'n'

	# Repeat script until user says is correct
	while correct != 'Y':

		# Get five points from image
		points = get_points(image)

		# Get coefficients for conic section from points
		cs = fivePointsToConic(points)

		# To store mask
		array = []

		# Get dimensions of mask
		dims = image.size

		# Finding out whether a point will be part of the masks
		for y in range(dims[1]):

			# Storing each row
			row = []

			# Generating row
			for x in range(dims[0]):

				# Left hand side of ellipse equation
				LHS = cs['a']*(x**2) + cs['b']*(x*y) + cs['c']*(y**2) + cs['d']*(x) + cs['e']*(y)

				# Right hand side of ellipse equation
				RHS = cs['f']

				# Checking if the point is outside the mask
				if LHS <= RHS:

					# Storing not in mask
					row.append(0)

				# If the point is in the mask
				else:

					# Storing in the mask
					row.append(1)

			# Adding row to array
			array.append(np.asarray(row))

		# Converting array to numpy array
		array = (np.asarray(array))

		# Generate mask
		mask = Image.fromarray(np.uint8(array * 255), 'L')

		# Displaying results
		f, axarr = plt.subplots(2)
		axarr[0].imshow(mask)
		axarr[1].imshow(image)

		# Show image
		plt.show()

		# Clear plots
		plt.clf()

		# Flag to store if correct
		correct = input('Does this look correct? [Y/n]. Input [invert] if you want to invert and save mask: ')

		if correct == 'invert':

			maskInvert = np.copy(mask)

			for i in range(len(mask)):

				for j in range(len(mask[0])):

					if mask[i][j] == 0:

						maskInvert[i][j] = 0

					else:

						maskInvert[i][j] = 255

	# Save mask

	if correct != 'invert':
		mask.save(outPath)
	else:
		maskInvert.save(outPath)

## Generate masks from cropped images
def generate_mask_all(outPath):

	# Path to access cropped images
	croppedPath = outPath + 'cropped/'

	# Path to save masks
	maskPath = outPath + 'masks/'

	# Create dir to store masks
	if len(glob.glob(maskPath)) == 0:
		os.mkdir(maskPath)

	croppedIms = glob.glob(croppedPath + '*')

	nCropped = len(croppedIms)

	print('\nFound ' + str(nCropped) + ' cropped images.')

	for cropped_im in croppedIms:

		imageName = cropped_im.split('\\')[-1].split('.')[-2]

		maskName = maskPath + imageName + '.png'

		print()
		print(cropped_im, maskName)

		if len(glob.glob(maskName)) == 0:
			generate_mask(cropped_im, maskName)
		else:
			print('Already exists.')

## Crop out bounding boxes from one image
def crop_bb(croppedPath, imPath, labelPath):

	# Prefix to cropped image
	prefix = croppedPath + imPath.split('/')[-1].split('.')[0] + '_'

	# Load labels for the image
	labels = np.loadtxt(labelPath)

	# How many cropped bounding boxes are there
	nCropped = len(labels)

	# Fill to output
	fill = len(str(nCropped))

	# Open image
	im = Image.open(imPath)

	# Get image dimensions
	width, height = im.size

	# For every image to be cropped (possibly)
	for j in range(nCropped):

		# Do only one in ninety-seven crops
		if j%97 == 0:

			# Name of cropped image
			outCropped = prefix + str(j).zfill(fill) + '.tif'

			# Extract particular line
			line = labels[j]

			# Format of label
			# class x_center y_center width height confidence

			# X center ratio
			x_c = line[1]
			# Y center ratio
			y_c = line[2]

			# Width of bounding box ratio
			w = line[3]
			# Height of bounding box ratio
			h = line[4]

			# Semi width in pixels
			semi_w = int(width*w/2)
			# Semi height in pixels
			semi_h = int(height*h/2)
			 
			# Setting the points for cropped image
			left = int(x_c*width) - semi_w
			top = int(y_c*height) - semi_h
			right = int(x_c*width) + semi_w
			bottom = int(y_c*height) + semi_h
			 
			# Cropped image
			cropped = im.crop((left, top, right, bottom))

			# Save cropped image
			cropped.save(outCropped)

## Crop out bounding boxes from images
def crop_bb_all(dirPath):

	imageDir = directory + 'images/'
	labelDir = directory + 'labels/'

	outputDir = 'output/'

	if len(glob.glob(outputDir)) == 0:

		os.mkdir(outputDir)

	outDir = outputDir + dirPath.split('/')[-2] + '/'

	if len(glob.glob(outDir)) != 0:

		shutil.rmtree(outDir)

	os.mkdir(outDir)

	croppedDir = outDir + 'cropped/'

	os.mkdir(croppedDir)

	nIm = len(glob.glob(imageDir + '*'))
	nLb = len(glob.glob(labelDir + '*'))

	if nIm != nLb:

		print('\nFound different numbers of images and labels.')

	for i in range(smaller(nIm, nLb)):

		# Do only one in eleven images
		if i%11 == 0:

			print(str(100*i/smaller(nIm, nLb))[:4] + '%')

			fId = str(i).zfill(4)

			imagePath = imageDir + fId + '.tif'
			labelPath = labelDir + fId + '.txt'

			if len(glob.glob(imagePath)) == 1 and len(glob.glob(labelPath)) == 1:

				crop_bb(croppedDir, imagePath, labelPath)
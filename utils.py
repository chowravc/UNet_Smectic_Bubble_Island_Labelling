## Importing packages
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import shutil


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
def generate_mask(path):

	# Path to store masks
	maskDirPath = 'output/' + path.split('/')[-2] + '/masks/'

	# Path to store image
	imageName = maskDirPath + path.split('/')[-1]

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
		correct = input('Does this look correct? [Y/n]: ')

	# Save mask
	mask.save(imageName)


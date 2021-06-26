import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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


def main():

	p1 = [20, 30]
	p2 = [30, 56]
	p3 = [60, 80]
	p4 = [20, 60]
	p5 = [400, 260]

	points = [p1, p2, p3, p4, p5]
	points = np.array(points)

	cs = fivePointsToConic(points)

	array = []

	dims = (512, 512)

	for x in range(dims[0]):

		row = []

		for y in range(dims[1]):

			LHS = cs['a']*(x**2) + cs['b']*(x*y) + cs['c']*(y**2) + cs['d']*(x) + cs['e']*(y)
			RHS = cs['f']

			if LHS <= RHS:

				row.append(0)

			else:

				row.append(1)

		array.append(np.asarray(row))

	array = (np.asarray(array))

	im = Image.fromarray(np.uint8(array * 255), 'L')

	plt.imshow(im)
	plt.show()

if __name__ == '__main__':

	# main()

	# Implementation of matplotlib function

	im = Image.open('face.jpg')

	plt.imshow(im)

	print("After 5 clicks :")
	x = plt.ginput(5)
	print(x)

	plt.show()
import math
import random
import numpy as np
import matplotlib.pyplot as plt

def trisample(A, B, C):
    """
    Given three vertices A, B, C, 
    sample point uniformly in the triangle
    """
    r1 = random.random()
    r2 = random.random()

    s1 = math.sqrt(r1)

    x = A[0] * (1.0 - s1) + B[0] * (1.0 - r2) * s1 + C[0] * r2 * s1
    y = A[1] * (1.0 - s1) + B[1] * (1.0 - r2) * s1 + C[1] * r2 * s1

    return (x, y)

random.seed(312345)
A = (1, 1)
B = (2, 4)
C = (5, 2)

def diff(x, y):

    dist = np.sqrt( (x[0] - y[0])**2 + (x[1] - y[1])**2 )

    return dist

nsample = 100000

points = []
label = []
for i in range(nsample):
    point = trisample(A, B, C)
    for c, j in enumerate([A,B,C]):
        dist = diff(point, j)
        if dist < 1:
            points.append(point)
            label.append(c)
            break

xx, yy = zip(*points)
plt.scatter(xx, yy, c=label, s=0.2, cmap="tab10")
plt.title('True latent variables')
plt.xlabel('z1')
plt.ylabel('z2')
plt.show()


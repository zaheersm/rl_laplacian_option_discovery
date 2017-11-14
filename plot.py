import numpy as np
import matplotlib.pyplot as plt
import itertools
'''
r = 3
c = 4
x = np.linspace(0, c, c+1)
y = np.linspace(0, r, r+1)

pts = itertools.product(x, y)
plt.scatter(*zip(*pts), marker='o', s=30, color='red')

X, Y = np.meshgrid(x, y)
deg = np.arctan(Y**3 - 3*Y-X)
QP = plt.quiver(X, Y, np.cos(deg), np.sin(deg))
plt.grid()
plt.show()
'''

X, Y = np.mgrid[0:5, 0:5]

U = []
V = []
Q = []

for i in range(5):

    t_u = []
    t_v = []
    t_q = []
    for j in range(5):
        state = np.array( X[i][j], Y[i][j])
        

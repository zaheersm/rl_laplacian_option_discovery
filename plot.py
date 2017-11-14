import numpy as np
import matplotlib.pyplot as plt

max_row = 10
max_col = 10
pi = np.zeros(max_row * max_col)

# check X,Y correspondence with row, col
Col, Row = np.meshgrid(range(0,max_col,1), range(0,max_row,1))
print(Row)
print(Col)

# U: x-component of vector
# V: y-component of vector
U = np.ones(max_row * max_col)
V = np.zeros(max_row * max_col)

Q = plt.quiver(Col, Row, U, V, pivot='mid', units='xy')

plt.xticks(range(max_col))
plt.yticks(range(max_row))
plt.gca().invert_yaxis()
plt.show()

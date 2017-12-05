import numpy as np
import matplotlib.pyplot as plt

import sys

def plot_pi(pi, max_row, max_col, display=True, savename=''):
    plt.figure()
    # Here we modify Terminate (0,0) to indicate termination with a dot
    # Note these actions denotes direction of (x,y) vector arrow and not (row,col)
    # Right, Left, Down, Up, Terminate
    action_set = [(1,0), (-1,0), (0,-1), (0,1), (0, 0)]
    vector_pi = [action_set[i] for i in pi]
   
    # Create grid
    Col, Row = np.meshgrid(range(0,max_col,1), range(0,max_row,1))

    # U: x-component of vector
    # V: y-component of vector
    U = []
    V = []

    for action in vector_pi:
        U.append(action[0])
        V.append(action[1])
    plt.figure()
    Q = plt.quiver(Col, Row, U, V, pivot='mid', units='xy', scale=3)

    plt.xticks(range(max_col))
    plt.yticks(range(max_row))
    plt.gca().invert_yaxis() # enable row,col indexing

    if savename != '':
        plt.savefig(savename)
    if display:
        plt.show()
    plt.close()
    
def pprint_pi(pi, max_row, max_col):
    action_set = ['R', 'L', 'D', 'U', 'T']
    count = 0
    for r in range((max_row)):
        for c in range((max_col)):
            sys.stdout.write(action_set[pi[count]] + ' ')
            count+=1
        print '\n'

def print_eigen(v, max_row, max_col):
    print
    count = 0
    for r in range(max_row):
        for c in range(max_col):
            if v[count] < 0:
                val = '{:3.2f}'.format(v[count])
            else:
                val = ' {:3.2f}'.format(v[count])
            sys.stdout.write(str(val) + ' ')
            count+=1
        print

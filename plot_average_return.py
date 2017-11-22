import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
   Results = np.load('ave_return.npy')

   plt.show()
   x_legend = range(len(Results[0][:]))
   graph_agent_1, = plt.plot(x_legend, Results[0][:], label="primitive actions")	
   graph_agent_2, = plt.plot(x_legend, Results[1][:], label="1 option")
   graph_agent_3, = plt.plot(x_legend, Results[2][:], label="2 options")
   graph_agent_4, = plt.plot(x_legend, Results[4][:], label="4 options")
   # graph_agent_5, = plt.plot(x_legend, Results[8][:], label="8 options")

   plt.legend(handles=[graph_agent_1, graph_agent_2, graph_agent_3, graph_agent_4]) #, graph_agent_5])
   #plt.xticks([0,8,18,28,38,48],[2,10,20,30,40,50])
   #plt.yticks([14,200,400,600,800])
   plt.xlabel('Episodes')
   plt.ylabel('Average return')
   plt.show()

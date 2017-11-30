import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
   Results = np.load('ave_return.npy')

   plt.show()
   x_legend = range(len(Results[0][:]))
   graph_agent_0, = plt.plot(x_legend, Results[0][:], label="primitive actions")	
   graph_agent_2, = plt.plot(x_legend, Results[2][:], label="2 option")
   graph_agent_4, = plt.plot(x_legend, Results[4][:], label="4 options")
   graph_agent_8, = plt.plot(x_legend, Results[8][:], label="8 options")
   graph_agent_64, = plt.plot(x_legend, Results[64][:], label="64 options")
   graph_agent_128, = plt.plot(x_legend, Results[128][:], label="128 options")
   graph_agent_200, = plt.plot(x_legend, Results[200][:], label="200 options")



   plt.legend(handles=[graph_agent_0, graph_agent_2, graph_agent_4, graph_agent_8, graph_agent_64, graph_agent_128, graph_agent_200])
   plt.xlabel('Episodes')
   plt.ylabel('Average return')
   plt.show()

import numpy as np
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.yticks(np.arange(0,20,1))
plt.show()

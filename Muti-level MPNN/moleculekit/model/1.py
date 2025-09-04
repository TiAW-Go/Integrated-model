import numpy as np

##设置全部数据，不输出省略号
import sys

np.set_printoptions(threshold=sys.maxsize)

data = np.load('qm8.npy')
print(data)
np.savetxt('data.txt', data, fmt='%s', newline='\n')
print('---------------------data--------------------------')
import numpy as np
import pandas as pd
import csv
array = []
data = open(r'data.txt')
for i in data:
    a = [float(x) for x in i.split()]
    array.append(a)

data.close()

with open("2.csv","a",newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(array)
    csvfile.close()

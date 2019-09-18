import os,sys
import h5py
import numpy as np
import networkx as nx
from pandas.core.frame import DataFrame
import pandas as pd

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from pylab import *
import math

outside_ids = [9, 10, 12, 13, 16, 17,
               19, 20, 21, 22, 23, 24, 37,
               39, 42, 46, 71, 72,
               82, 85, 96, 98, 101, 103, 111,
               113, 114, 121,
               122, 126, 127, 128, 130, 131,
               132, 134]

for j in range(len(outside_ids)):
    SKEL_FOLDER = '/home/xingyu/PycharmProjects/skeleton/outputs/'+str(outside_ids[j])
    data = pd.read_csv(os.path.join(SKEL_FOLDER,'analysis.csv'))
    num = len(data['rank'].unique())
    distance_all =[]
    objects = []
    for i in range(num-1):
        distance_i = data[data['rank']==i+1]
        distance_i = np.array(distance_i['distance'].tolist())
        distance_i = distance_i[np.argsort(distance_i)]
        distance_all.extend(distance_i.tolist())
        name = 'Rank'+str(i+1)
        objects +=  [name]*len(distance_i)


    s = distance_all
    t = np.arange(len(s))



    plt.figure(0)
    plt.title('Rank Distance')
    plt.bar(t, s, width = 0.8, align='center', alpha=0.5)
    plt.xticks(t, objects)
    plt.ylabel('Distance')


    plt.savefig(os.path.join(SKEL_FOLDER,'distance_bar.png'), dpi=100)
    plt.close(0)

    print('finshed:', j, '/', len(outside_ids))

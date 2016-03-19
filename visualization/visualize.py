import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.optimizers import SGD, RMSprop
from collections import deque
import os
import glob
import numpy as np
import queue
import threading
import time
import random
import math
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('path', type=str)
args = parser.parse_args()

directory = args.path
model = model_from_json(open(max(glob.iglob(directory + '/model_*.json'), key=os.path.getctime)).read())
model.load_weights(max(glob.iglob(directory + '/weights_*.h5'), key=os.path.getctime))
minimum = min([np.amin(a) for a in model.get_weights()])

plt.axis('off')
for i in range(0,32):
    frame = plt.subplot(8,4,i+1)
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.pcolor((model.get_weights()[0][i][0] + minimum), cmap=plt.get_cmap('Greys'))

plt.tight_layout()
plt.savefig(directory + '/conv_weights.png')

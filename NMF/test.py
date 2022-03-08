import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv', sep='\t', header=None).values
number1 = 255 * data[:, 0].reshape(64, 64).T
img1 = Image.fromarray(number1)
img1.show()
print(number1.shape)





import os
import sys

# Disable tensorflow compilation warnings
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from skimage import io

img = io.imread("Man.jpg", as_gray=True)

print(img)
img.show()



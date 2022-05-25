import matplotlib.pyplot as plt
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

#Heatmap means color map
language = 'python', 'java', 'PHP', 'HTML', 'JavaScript', 'C++'
popularity = [22.2, 17, 15.5, 8, 9, 3]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c","blue", "red", "green"]
explode = (0.1, 0,0,0,0,0)
plt.pie(popularity, explode= explode, labels=language, colors=colors, autopct='%1.1f%%',shadow=True, startangle=140)
#plt.show()





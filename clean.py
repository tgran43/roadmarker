from random import sample
import os

count = 0
files = os.listdir('D:/output/val/class2')
for file in sample(files,657 - 86):
    os.remove('D:/output/val/class2/' + file)
    count+=1
print(count)

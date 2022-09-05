gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import os
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
dataset_path = '/content/gdrive/MyDrive/DataSets/NUS-Hand-Posture-Dataset-II-tbu'
import pandas as pd
from matplotlib.pyplot import imread
uniques = [ "a" , "b" , "c" , "d", "e", "f", "g", "h", "i", "j"]
dirs = ["Train"]
data = []
for dir in dirs :
  for unique in uniques:
    directory = "/content/gdrive/MyDrive/DataSets/NUS-Hand-Posture-Dataset-II-tbu/" + dir + "/" + unique

    for filename in os.listdir(directory):
      path = directory + "/" + filename 
      data.append([ filename , path  , unique])
df = pd.DataFrame(data, columns = ["filename" ,"path", "class"])
y = np.array([i for i in df["class"]])
x = np.array([i for i in df["path"]])
def encode_y(y):
  Y = []
  for i in y : 
    if(i == "a" ):
      Y.append(0)
    if(i == "b" ):
      Y.append(1)
    if(i == "c" ):
      Y.append(2)
    if(i == "d" ):
      Y.append(3)
    if(i == "e" ):
      Y.append(4)
    if(i == "f" ):
      Y.append(5)
    if(i == "g" ):
      Y.append(6)
    if(i == "h" ):
      Y.append(7)
    if(i == "i" ):
      Y.append(8)
    if(i=="j" ):
      Y.append(9)  
  return  np.array(Y).astype("float32")
  mean_size = (120, 160)
def feature_extraction(x, mean_size):
  Image = []
  for i in x:
    img = cv2.imread(i)
    resized_img = cv2.resize(img,mean_size)
    resized_normalized_img=resized_img/255.0;
    Image.append(resized_normalized_img)
  return np.array(Image)
X_train = feature_extraction(x, mean_size)
y_train = encode_y(y)
y_train = y_train.reshape(-1,1)
y_train = y_train.astype(int)
from tensorflow.python.keras.backend import dtype
def arr(y):
    two_d=np.zeros((len(y),10), dtype=float)
    for i in range(len(two_d)):
        two_d[i][y[i]]=1
    return two_d
y_train = arr(y_train)
import pandas as pd
from matplotlib.pyplot import imread
uniques = [ "a" , "b" , "c" , "d", "e", "f", "g", "h", "i", "j"]
dirs = ["Test"]
data = []
for dir in dirs :
  for unique in uniques:
    directory = "/content/gdrive/MyDrive/DataSets/NUS-Hand-Posture-Dataset-II-tbu/" + dir + "/" + unique

    for filename in os.listdir(directory):
      path = directory + "/" + filename 
      data.append([ filename , path  , unique])
df = pd.DataFrame(data, columns = ["filename" ,"path", "class"])
y = np.array([i for i in df["class"]])
x = np.array([i for i in df["path"]])
def encode_y(y):
  Y = []
  for i in y : 
    if(i == "a" ):
      Y.append(0)
    if(i == "b" ):
      Y.append(1)
    if(i == "c" ):
      Y.append(2)
    if(i == "d" ):
      Y.append(3)
    if(i == "e" ):
      Y.append(4)
    if(i == "f" ):
      Y.append(5)
    if(i == "g" ):
      Y.append(6)
    if(i == "h" ):
      Y.append(7)
    if(i == "i" ):
      Y.append(8)
    if(i=="j" ):
      Y.append(9)  
  return  np.array(Y).astype("float32")
X_test = feature_extraction(x, mean_size)
y_test = encode_y(y)
y_test = y_test.reshape(-1,1)
y_test = y_test.astype(int)
from tensorflow.python.keras.backend import dtype
def arr(y):
    two_d=np.zeros((len(y),10), dtype=float)
    for i in range(len(two_d)):
        two_d[i][y[i]]=1
    return two_d
y_test = arr(y_test)
image_size = (160,120)
batch_size = 16
train_datagen = ImageDataGenerator( width_shift_range=0.3,
                                    height_shift_range=0.3,
                                    zoom_range=0.3
                                    )
val_datagen = ImageDataGenerator()

train =  train_datagen.flow(X_train, y_train, 
                            batch_size=batch_size,
                            shuffle = True)
validation = val_datagen.flow(X_test ,y_test,
                          batch_size=batch_size,
                          shuffle = True)

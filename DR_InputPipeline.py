from google.colab import drive
from __future__ import absolute_import, division, print_function, unicode_literals

drive.mount('/content/drive')
!ls "/content/drive/My Drive"

!pip3 install tensorflow-gpu==2.0.0
!pip3 install grpcio
!pip install -q pyyaml h5py

import tensorflow  as tf
import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot as plt
import glob
import natsort
import pandas as pd
import cv2
import os

from google.colab.patches import cv2_imshow
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator
from tensorflow.python.keras.utils.data_utils import Sequence

N_epochs = 200
learning_rate = 0.001
N_batch_size = 32
N_parallel_iterations = 4
N_prefetch = 8
N_shuffle_buffer = 200
N_training_examples = 330
N_test_examples = 103
N_validation_examples = 83
N_image_size = 256

#Read Labels .csv file from Google Drive
with open("/content/drive/My Drive/IDRID_Disease_Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",newline='') as csvfile:
    df_train = pd.read_csv(csvfile)

with open("/content/drive/My Drive/IDRID_Disease_Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",newline='') as csvfile:
    df_test = pd.read_csv(csvfile)

df_train = df_train['Retinopathy grade'].values
df_test = df_test['Retinopathy grade'].values

#debug
#print(df_train)
#print(df_test)

#Function to encode Labels to 0 and 1 for binary classification
#Input: Label Values in range (0,4)
#Ouput: Label value encoded to 0 or 1 
def int_to_classes(i):
    if i == 0: return 0
    elif i == 1: return 0
    elif i == 2: return 1
    elif i == 3: return 1
    elif i == 4: return 1

#Encode Train and Test Labels to 0, 1
df_train = [int_to_classes(i) for i in df_train]
df_test = [int_to_classes(i) for i in df_test]

#debug
#print(df_train)
#print(df_test)

labels_train = np.asarray(df_train).astype('float32')
labels_test = np.asarray(df_test).astype('float32')
#print(len(labels_test))

#Read Training and Test Images which are in .jpg format
files_train = glob.glob("/content/drive/My Drive/IDRID_Disease_Grading/B. Disease Grading/1. Original Images/a. Training Set/*.jpg")
files_test = glob.glob("/content/drive/My Drive/IDRID_Disease_Grading/B. Disease Grading/1. Original Images/b. Testing Set/*.jpg")

#Sort the images in ascending order 
files_train = natsort.natsorted(files_train,reverse=False)
files_test = natsort.natsorted(files_test,reverse=False)

#debug
#print(files_train)
#print(files_test)

#debug
"""
image_string = tf.io.read_file(files_train[0])# Read the image
image_decoded = tf.io.decode_jpeg(image_string) # Decode the image
img=tf.image.rgb_to_grayscale(image_decoded)
img = tf.image.central_crop(img, 0.85)
print(img.shape)
img = tf.image.crop_to_bounding_box(img, offset_height = 0 , offset_width = 0 , target_height = 2422, target_width = 3380)
img = tf.image.resize(img, size=(N_image_size, N_image_size))
#img = np.asarray(img)

plt.imshow(np.squeeze(img))
#plt.imshow(img,cmap='gray')
plt.show()
"""

#Function to do preprocessing on the images (preprocessing includes crop and resize)
#Input: Image and its corresponding label
#Output: Preprocessed Image and its label
def parse_func(filename, label):

    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string)
    image_crop = tf.image.central_crop(image_decoded, 0.85)
    image_crop2 = tf.image.crop_to_bounding_box(image_crop,
                                                offset_height=0,
                                                offset_width=0,
                                                target_height=2422,
                                                target_width=3380)
    image_resize = tf.image.resize(image_crop2, size=(N_image_size, N_image_size))
    image = tf.reshape(image_resize, [-1])
    return image, label

#Function to build dataset
#Input: Image, Label pairs
#Output: Dataset with Image-Label pairs
def build_dataset(files, labels):

    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.shuffle(N_shuffle_buffer)
    ds = ds.map(parse_func, num_parallel_calls = N_parallel_iterations)
    ds = ds.prefetch(N_prefetch)
    return ds

#Build Train and Test Datasets
train_ds = build_dataset(
                         files_train[0:N_training_examples+N_validation_examples],
                         labels_train[0:N_training_examples+N_validation_examples]
                        )

test_ds = build_dataset(files_test[0:N_test_examples],labels_test[0:N_test_examples])

#Debug
"""
for image, label in train_ds.take(1):
  image_matrix = image.numpy()
  label_matrix = label.numpy()
  print(image_matrix.shape)
  #print(label_matrix)
"""

#Debug
"""
for image, label in test_ds.take(1):
  image_matrix = image.numpy()
  label_matrix = label.numpy()
  print(image_matrix.shape)
  #print(label_matrix)
"""

image_matrix_appended = []
label_matrix_appended = []

#Seperate out images and labels from the dataset and convert them into numpy array
#Do it for both Train and Test Dataset
for image, label in train_ds:
  image_matrix_appended.append(np.array(image).flatten().reshape(3, N_image_size * N_image_size))
  label_matrix_appended.append(np.array(label))

image_matrix_appended_test = []
label_matrix_appended_test = []

for image, label in test_ds:
  image_matrix_appended_test.append(np.array(image).flatten().reshape(3, N_image_size * N_image_size))
  label_matrix_appended_test.append(np.array(label))

#Debug
#print(len(image_matrix_appended))
#print(image_matrix_appended[0:2])
#print(len(label_matrix_appended))

#Debug
#print(len(image_matrix_appended_test))
#print(image_matrix_appended_test[0:2])
#print(len(label_matrix_appended_test))

##Debug
#print(labels_train)
#print(labels_test)

#Numpy array of all train/test images
image_matrix_appended = np.asarray(image_matrix_appended).astype('float32')
image_matrix_appended_test = np.asarray(image_matrix_appended_test).astype('float32')

#Numpy array of train/test labels
label_matrix_appended = np.asarray(label_matrix_appended).astype('float32')
label_matrix_appended_test = np.asarray(label_matrix_appended_test).astype('float32')

##Debug
'''
print(label_matrix_appended)
print(label_matrix_appended_test)
print(len(label_matrix_appended))
print(len(label_matrix_appended_test))
'''

image_matrix_appended = image_matrix_appended.reshape(image_matrix_appended.shape[0], N_image_size, N_image_size, 3)
image_matrix_appended_test = image_matrix_appended_test.reshape(image_matrix_appended_test.shape[0], N_image_size, N_image_size, 3)

#Normalize the images 
image_matrix_appended /= 255
#print(image_matrix_appended.shape)

image_matrix_appended_test /= 255
#print(image_matrix_appended_test.shape)

label_matrix_appended = label_matrix_appended.reshape(label_matrix_appended.shape[0], 1)
label_matrix_appended_test = label_matrix_appended_test.reshape(label_matrix_appended_test.shape[0], 1)

#debug
'''
print(label_matrix_appended)
print(label_matrix_appended_test)
print(len(label_matrix_appended))
print(len(label_matrix_appended_test))
'''

#This class implements function to remove imbalance in the dataset by oversampling the images of the class which has less images. 
#It generates images using ImageDataGenerator
#Output: Gives equal images for both classes. 
class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + Random_Over-sampling"""
    def __init__(self, x, y, datagen, batch_size):
        self.datagen = datagen
        self.batch_size = batch_size
        self._shape = x.shape
        datagen.fit(x)
        self.gen, self.steps_per_epochs = balanced_batch_generator(x.reshape(x.shape[0], -1),
                                                                   y,
                                                                   sampler = RandomOverSampler(),
                                                                   batch_size = self.batch_size,
                                                                   keep_sparse = True)
    def __len__(self):
        return self._shape[0] // self.batch_size

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch,
                                 y_batch,
                                 batch_size = self.batch_size).next()
    
#Augment Images using rotation, horizontal flip and zoom 
datagen = ImageDataGenerator(rotation_range = 10, zoom_range=0.001, horizontal_flip = True)

#Balance Train dataset using BalancedDataGenerator and datagen defined above
balanced_gen = BalancedDataGenerator(image_matrix_appended[0:N_training_examples],
                                     label_matrix_appended[0:N_training_examples],
                                     datagen,
                                     batch_size = N_batch_size)

#Number of steps per epoch = Dataset size divided by batch size
steps_per_epochs = balanced_gen.steps_per_epochs
print(steps_per_epochs)

#debug
'''
y_gen = [balanced_gen.__getitem__(0)[1] for i in range(steps_per_epochs)]
print(y_gen)
'''
#Print number of samples per class. They are now equal. Imbalance in dataset is removed
y_gen = [balanced_gen.__getitem__(0)[1] for i in range(steps_per_epochs)]
print(np.unique(y_gen, return_counts=True))

#Print sample image from the balanced dataset
y_gen = [balanced_gen.__getitem__(0)[0][0]]
y_gen = np.asarray(y_gen).reshape(N_image_size, N_image_size, 3)

plt.imshow(y_gen)
plt.show()

#Define Validation Dataset
X_val = image_matrix_appended[N_training_examples:N_training_examples + N_validation_examples]
Y_val = label_matrix_appended[N_training_examples:N_training_examples + N_validation_examples]

#No Augmentation for Validation Images
validationdatagenerator = ImageDataGenerator()
validation_generator = validationdatagenerator.flow(X_val,
                                                    Y_val,
                                                    batch_size = N_batch_size)
#Define Test Dataset; No Augmentation for Test Images 
testdatagenerator = ImageDataGenerator()
test_generator = testdatagenerator.flow(image_matrix_appended_test[0:N_test_examples],
                                        label_matrix_appended_test[0:N_test_examples],
                                        batch_size = 1)

#*Augmented Image Visualization*****************************************************************************************

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#Plot augmented images from train dataset
augmented_images = [balanced_gen[0][0][0] for i in range(10)]
plotImages(augmented_images)

#*Augmented Image Visualization ends************************************************************************************






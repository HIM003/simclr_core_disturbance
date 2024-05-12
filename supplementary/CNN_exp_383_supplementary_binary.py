#!/usr/bin/env python
# coding: utf-8

# # Workbook for Image Classfication using CNN

# In[1]:


import os
import zipfile
import random
import shutil
#import tensorflow as tf
#import pandas as pd
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
#import matplotlib.pyplot as plt
#from tensorflow.python.client import device_lib
#import numpy as np
#from keras.preprocessing import image
#tf.debugging.set_log_device_placement(True)

'''
# ## 1) Split the dataset into Training / Validation & Test 

# ##### How many images in each category?

# In[2]:


os.chdir(r'/rds/general/user/hm808/home/images')
source_path = 'for_training'
folders = os.listdir(source_path)
for i in range(len(folders)):
    source_path_cat = os.path.join(source_path,folders[i])
    print(f"There are {len(os.listdir(source_path_cat))} images of {folders[i]}.")


# ##### Create testing/training file structure

# In[3]:


# Define root directory
root_dir = r'/rds/general/user/hm808/home/images/tmp_binary_10train'
root_dir_up = r'/rds/general/user/hm808/home/images/for_training'

# Empty directory to prevent FileExistsError is the function is run several times
if os.path.exists(root_dir):
  shutil.rmtree(root_dir)


def create_train_test_dirs(root_dir, root_dir_up):
  os.makedirs(root_dir)
  os.makedirs(os.path.join(root_dir, "training"))
  os.makedirs(os.path.join(root_dir, "validation"))
  os.makedirs(os.path.join(root_dir, "testing"))
  folders = os.listdir(root_dir_up)
  for i in range(len(folders)):
    os.makedirs(os.path.join(root_dir,"training",folders[i]))
    os.makedirs(os.path.join(root_dir,"validation",folders[i]))
    os.makedirs(os.path.join(root_dir,"testing",folders[i]))
 
try:
  create_train_test_dirs(root_dir, root_dir_up)
except FileExistsError:
  None 


# ##### Split the images randomly into test and train folders based on split size 

# In[4]:


def split_data(SOURCE, TRAINING, VALIDATION, TESTING, SPLIT_SIZE_TRAIN, SPLIT_SIZE_VAL):

  files = random.sample(os.listdir(SOURCE), len(os.listdir(SOURCE)))
  for i in range(len(files)):
    source_file = os.path.join(SOURCE, files[i])
    train_file = os.path.join(TRAINING, files[i])
    validation_file = os.path.join(VALIDATION, files[i])
    test_file = os.path.join(TESTING, files[i])
    
    if os.path.getsize(source_file) == 0:
      print(files[i], " is zero length, so ignoring.")
    else:
      if i <= int(len(files)*SPLIT_SIZE_TRAIN):
          copyfile(source_file, train_file)
      
      elif  (i > int(len(files)*SPLIT_SIZE_TRAIN)) & (i <= int(len(files)*SPLIT_SIZE_VAL)):
          copyfile(source_file, validation_file)
    
      else:
          copyfile(source_file, test_file)


# In[5]:


root_path = r"/rds/general/user/hm808/home/images"
source_ext = r"for_training"
source_path = os.path.join(root_path, source_ext)
test_ext = r"tmp_binary_10train/testing"
validation_ext = r"tmp_binary_10train/validation"
train_ext = r"tmp_binary_10train/training"
split_size_train = 0.1
split_size_validation = 0.15


folders = os.listdir(source_path)
for i in range(len(folders)):
    source_path_ = os.path.join(root_path,source_ext,folders[i])
    TESTING_ = os.path.join(root_path,test_ext,folders[i])
    VALIDATION_ = os.path.join(root_path,validation_ext,folders[i])
    TRAINING_ = os.path.join(root_path,train_ext,folders[i])
    split_data(source_path_, TRAINING_, VALIDATION_, TESTING_, split_size_train, split_size_validation)


# ##### Number of files in training and testing after random split

# In[6]:


os.chdir(r'/rds/general/user/hm808/home/images/tmp_binary_10train')
train_path = 'training'
validation_path = 'validation'
test_path = 'testing'

folders = os.listdir(train_path)
for i in range(len(folders)):
    train_path_cat = os.path.join(train_path,folders[i])
    validation_path_cat = os.path.join(validation_path,folders[i])
    test_path_cat = os.path.join(test_path,folders[i])
    print(f"{folders[i]}: {len(os.listdir(train_path_cat))} training & {len(os.listdir(validation_path_cat))} validation & {len(os.listdir(test_path_cat))} testing ")


quit()

'''
# ## 2) Split the dataset into Training / Validation & Test (Binary)

# ##### How many images in each category?

# In[3]:


os.chdir(r'/jmain02/home/J2AD015/axf03/hxm18-axf03/images')
source_path = 'for_training'
folders = os.listdir(source_path)
for i in range(len(folders)):
    source_path_cat = os.path.join(source_path,folders[i])
    print(f"There are {len(os.listdir(source_path_cat))} images of {folders[i]}.")


# ##### Create testing/training file structure

# In[5]:


# Define root directory
root_dir = r'/rds/general/user/hm808/home/images/tmp_binary_10train_fold1'
root_dir_up = r'/rds/general/user/hm808/home/images/for_training'

# Empty directory to prevent FileExistsError is the function is run several times
if os.path.exists(root_dir):
  shutil.rmtree(root_dir)


def create_train_test_dirs(root_dir, root_dir_up):
  os.makedirs(root_dir)
  os.makedirs(os.path.join(root_dir, "training"))
  os.makedirs(os.path.join(root_dir, "validation"))
  os.makedirs(os.path.join(root_dir, "testing"))
  folders = list(["none", "disturbance"])
  for i in range(len(folders)):
    os.makedirs(os.path.join(root_dir,"training",folders[i]))
    os.makedirs(os.path.join(root_dir,"validation",folders[i]))
    os.makedirs(os.path.join(root_dir,"testing",folders[i]))
 
try:
  create_train_test_dirs(root_dir, root_dir_up)
except FileExistsError:
  None 


# ##### Split the images randomly into test and train folders based on split size 

# In[7]:


def split_data(SOURCE, TRAINING, VALIDATION, TESTING, SPLIT_SIZE_TRAIN, SPLIT_SIZE_VAL):

  files = random.sample(os.listdir(SOURCE), len(os.listdir(SOURCE)))
  for i in range(len(files)):
    source_file = os.path.join(SOURCE, files[i])
    train_file = os.path.join(TRAINING, files[i])
    validation_file = os.path.join(VALIDATION, files[i])
    test_file = os.path.join(TESTING, files[i])
    
    if os.path.getsize(source_file) == 0:
      print(files[i], " is zero length, so ignoring.")
    else:
      if i <= int(len(files)*SPLIT_SIZE_TRAIN):
          copyfile(source_file, train_file)
      
      elif  (i > int(len(files)*SPLIT_SIZE_TRAIN)) & (i <= int(len(files)*SPLIT_SIZE_VAL)):
          copyfile(source_file, validation_file)
    
      else:
          copyfile(source_file, test_file)


# In[9]:


root_path = r"/jmain02/home/J2AD015/axf03/hxm18-axf03/images"
source_ext = r"for_training"
source_path = os.path.join(root_path, source_ext)
test_ext = r"tmp_binary_10train_fold1/testing"
validation_ext = r"tmp_binary_10train_fold1/validation"
train_ext = r"tmp_binary_10train_fold1/training"
split_size_train = 0.1
split_size_validation = 0.15


folders = os.listdir(source_path)
for i in range(len(folders)):
    source_path_ = os.path.join(root_path,source_ext,folders[i])
    if folders[i] == "none":
        TESTING_ = os.path.join(root_path,test_ext,folders[i])
        VALIDATION_ = os.path.join(root_path,validation_ext,folders[i])
        TRAINING_ = os.path.join(root_path,train_ext,folders[i])
    else:
        TESTING_ = os.path.join(root_path,test_ext,"disturbance")
        VALIDATION_ = os.path.join(root_path,validation_ext,"disturbance")
        TRAINING_ = os.path.join(root_path,train_ext,"disturbance")       
    split_data(source_path_, TRAINING_, VALIDATION_, TESTING_, split_size_train, split_size_validation)


# ##### Number of files in training and testing after random split

# In[10]:


os.chdir(r'/jmain02/home/J2AD015/axf03/hxm18-axf03/images/tmp_binary_10train_fold1')
train_path = 'training'
validation_path = 'validation'
test_path = 'testing'

folders = os.listdir(train_path)
for i in range(len(folders)):
    train_path_cat = os.path.join(train_path,folders[i])
    validation_path_cat = os.path.join(validation_path,folders[i])
    test_path_cat = os.path.join(test_path,folders[i])
    print(f"{folders[i]}: {len(os.listdir(train_path_cat))} training & {len(os.listdir(validation_path_cat))} validation & {len(os.listdir(test_path_cat))} testing ")


quit()
# ## 3) Split the dataset into Training / Validation & Test (Multi-Class re-combined)

# ##### How many images in each category?

# In[2]:


os.chdir(r'D:\OneDrive\Documents\Imperial\Data\images\383\drill_disturbances\manual')
source_path = 'for_training'
folders = os.listdir(source_path)
for i in range(len(folders)):
    source_path_cat = os.path.join(source_path,folders[i])
    print(f"There are {len(os.listdir(source_path_cat))} images of {folders[i]}.")


# ##### Create testing/training file structure

# In[16]:


# Define root directory
root_dir = r'D:\OneDrive\Documents\Imperial\Data\images\383\drill_disturbances\manual\tmp_mult_recom'
root_dir_up = r'D:\OneDrive\Documents\Imperial\Data\images\383\drill_disturbances\manual\for_training'

# Empty directory to prevent FileExistsError is the function is run several times
if os.path.exists(root_dir):
  shutil.rmtree(root_dir)


crack_frac_brec = list(['crack', 'brecciated', 'fractured'])
fall_flow_suck = list(['fall-in', 'flow-in', 'suck-in'])

def create_train_test_dirs(root_dir, root_dir_up):
  os.makedirs(root_dir)
  os.makedirs(os.path.join(root_dir, "training"))
  os.makedirs(os.path.join(root_dir, "validation"))
  os.makedirs(os.path.join(root_dir, "testing"))
  folders = os.listdir(root_dir_up)
  for i in range(len(folders)):
    if not folders[i] in crack_frac_brec + fall_flow_suck:
        os.makedirs(os.path.join(root_dir,"training",folders[i]))
        os.makedirs(os.path.join(root_dir,"validation",folders[i]))
        os.makedirs(os.path.join(root_dir,"testing",folders[i]))
    else:
        None
  os.makedirs(os.path.join(root_dir,"training","crack_frac_brec"))
  os.makedirs(os.path.join(root_dir,"validation","crack_frac_brec"))
  os.makedirs(os.path.join(root_dir,"testing","crack_frac_brec"))
  os.makedirs(os.path.join(root_dir,"training","fall_flow_suck"))
  os.makedirs(os.path.join(root_dir,"validation","fall_flow_suck"))
  os.makedirs(os.path.join(root_dir,"testing","fall_flow_suck"))

        
try:
    #None
    create_train_test_dirs(root_dir, root_dir_up)
except FileExistsError:
    None 


# ##### Split the images randomly into test and train folders based on split size 

# In[17]:


def split_data(SOURCE, TRAINING, VALIDATION, TESTING, SPLIT_SIZE_TRAIN, SPLIT_SIZE_VAL):

  files = random.sample(os.listdir(SOURCE), len(os.listdir(SOURCE)))
  for i in range(len(files)):
    source_file = os.path.join(SOURCE, files[i])
    train_file = os.path.join(TRAINING, files[i])
    validation_file = os.path.join(VALIDATION, files[i])
    test_file = os.path.join(TESTING, files[i])
    
    if os.path.getsize(source_file) == 0:
      print(files[i], " is zero length, so ignoring.")
    else:
      if i <= int(len(files)*SPLIT_SIZE_TRAIN):
          copyfile(source_file, train_file)
      
      elif  (i > int(len(files)*SPLIT_SIZE_TRAIN)) & (i <= int(len(files)*SPLIT_SIZE_VAL)):
          copyfile(source_file, validation_file)
    
      else:
          copyfile(source_file, test_file)


# In[18]:


root_path = r"D:\OneDrive\Documents\Imperial\Data\images\383\drill_disturbances\manual"
source_ext = r"for_training"
source_path = os.path.join(root_path, source_ext)
test_ext = r"tmp_mult_recom\testing"
validation_ext = r"tmp_mult_recom\validation"
train_ext = r"tmp_mult_recom\training"
split_size_train = 0.7
split_size_validation = 0.9

crack_frac_brec = list(['crack', 'brecciated', 'fractured'])
fall_flow_suck = list(['fall-in', 'flow-in', 'suck-in'])


folders = os.listdir(source_path)
for i in range(len(folders)):
    if folders[i] in crack_frac_brec:
        source_path_ = os.path.join(root_path,source_ext,folders[i])
        TESTING_ = os.path.join(root_path,test_ext,"crack_frac_brec")
        VALIDATION_ = os.path.join(root_path,validation_ext,"crack_frac_brec")
        TRAINING_ = os.path.join(root_path,train_ext,"crack_frac_brec")
        split_data(source_path_, TRAINING_, VALIDATION_, TESTING_, split_size_train, split_size_validation)
    elif folders[i] in fall_flow_suck:
        source_path_ = os.path.join(root_path,source_ext,folders[i])
        TESTING_ = os.path.join(root_path,test_ext,"fall_flow_suck")
        VALIDATION_ = os.path.join(root_path,validation_ext,"fall_flow_suck")
        TRAINING_ = os.path.join(root_path,train_ext,"fall_flow_suck")
        split_data(source_path_, TRAINING_, VALIDATION_, TESTING_, split_size_train, split_size_validation)
    else:
        source_path_ = os.path.join(root_path,source_ext,folders[i])
        TESTING_ = os.path.join(root_path,test_ext,folders[i])
        VALIDATION_ = os.path.join(root_path,validation_ext,folders[i])
        TRAINING_ = os.path.join(root_path,train_ext,folders[i])
        split_data(source_path_, TRAINING_, VALIDATION_, TESTING_, split_size_train, split_size_validation)


# ##### Number of files in training and testing after random split

# In[2]:


os.chdir(r'D:\OneDrive\Documents\Imperial\Data\images\383\drill_disturbances\manual\tmp_mult_recom')
train_path = 'training'
validation_path = 'validation'
test_path = 'testing'

folders = os.listdir(train_path)
for i in range(len(folders)):
    train_path_cat = os.path.join(train_path,folders[i])
    validation_path_cat = os.path.join(validation_path,folders[i])
    test_path_cat = os.path.join(test_path,folders[i])
    print(f"{folders[i]}: {len(os.listdir(train_path_cat))} training & {len(os.listdir(validation_path_cat))} validation & {len(os.listdir(test_path_cat))} testing ")


# ## 4) Supplementary

# ### Evaluating Individual Predictions

# In[21]:


# class names
image_dir = r"D:\OneDrive\Documents\Imperial\Data\images\383\drill_disturbances\manual\tmp\testing"
image_fnames = os.listdir(image_dir)
image_fnames

# file names
fname = []
for root,d_names,f_names in os.walk(image_dir):
    for f in f_names:
        fname.append(os.path.join(root, f))

#select random image and compare label to predicted
fname_rand = random.choice(fname)
img = image.load_img(fname_rand, target_size=(150, 150))
x=image.img_to_array(img)
x /= 255
x=np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10) 
plt.imshow(img)
plt.title("Label:{} \n Predicted:{}".format(fname_rand.split("\\")[-2],image_fnames[np.argmax(classes[0])]))


# ### Visualizing Intermediate Representations
# 

# In[17]:


import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model
successive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

suck_in_dir = r"D:\OneDrive\Documents\Imperial\Data\images\383\drill_disturbances\manual\tmp\testing\suck-in"
soupy_dir = r"D:\OneDrive\Documents\Imperial\Data\images\383\drill_disturbances\manual\tmp\testing\soupy"
suck_in_fnames = os.listdir( suck_in_dir )
soupy_fnames = os.listdir( soupy_dir )



# Prepare a random input image from the training set.
suck_in_img_files = [os.path.join(suck_in_dir, f) for f in suck_in_fnames]
soupy_img_files = [os.path.join(soupy_dir, f) for f in soupy_fnames]
img_path = random.choice(suck_in_img_files + soupy_img_files)
print(img_path)
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Scale by 1/255
x /= 255.0

# Run the image through the network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so you can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Display the representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
  if len(feature_map.shape) == 4:
    
    #-------------------------------------------
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    #-------------------------------------------
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
    # Tile the images in this matrix
    display_grid = np.zeros((size, size * n_features))
    
    #-------------------------------------------------
    # Postprocess the feature to be visually palatable
    #-------------------------------------------------
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    #-----------------
    # Display the grid
    #-----------------
    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 


# You can see above how the pixels highlighted turn to increasingly abstract and compact representations, especially at the bottom grid. 
# 
# The representations downstream start highlighting what the network pays attention to, and they show fewer and fewer features being "activated"; most are set to zero. This is called _representation sparsity_ and is a key feature of deep learning. These representations carry increasingly less information about the original pixels of the image, but increasingly refined information about the class of the image. You can think of a convnet (or a deep network in general) as an information distillation pipeline wherein each layer filters out the most useful features.

# ### Different Methods to check with GPU working

# In[2]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print(tf.test.is_built_with_cuda())
tf.config.list_physical_devices('GPU') 
print(device_lib.list_local_devices())


# ### Callbacks

# In[63]:


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    '''
    Halts the training after reaching 60 percent accuracy

    Args:
      epoch (integer) - index of epoch (required but unused in the function definition below)
      logs (dict) - metric results from the training epoch
    '''

    # Check accuracy
    if(logs.get('loss') < 0.4):

      # Stop if threshold is met
      print("\nLoss is lower than 0.4 so cancelling training!")
      self.model.stop_training = True

# Instantiate class
callbacks = myCallback()

# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True


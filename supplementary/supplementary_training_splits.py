import sys
import os
import zipfile
import random
import shutil
from shutil import copyfile


path1 = sys.argv[1] #for_training
path2 = sys.argv[2] #for_training_20
split_size = float(sys.argv[3]) #0.2
source_path = '/jmain02/home/J2AD015/axf03/hxm18-axf03/images'
root_dir_up = os.path.join(source_path, path1)
root_dir = os.path.join(source_path, path2)


### Split the dataset into Training / Validation & Test (Binary)
###### How many images in each category?
os.chdir(source_path)
folders = os.listdir(root_dir_up)
for i in range(len(folders)):
    source_path_cat = os.path.join(root_dir_up,folders[i])
    print(f"There are {len(os.listdir(source_path_cat))} images of {folders[i]}.")


# ##### Create testing/training file structure
# Empty directory to prevent FileExistsError is the function is run several times
if os.path.exists(root_dir):
  shutil.rmtree(root_dir)

def create_train_test_dirs(root_dir):
  os.makedirs(root_dir)
  for i in range(len(folders)):
    os.makedirs(os.path.join(root_dir,folders[i]))
 
try:
  create_train_test_dirs(root_dir)
except FileExistsError:
  None 


# ##### Split the images randomly into test and train folders based on split size 
def split_data(SOURCE, TRAINING, SPLIT_SIZE_TRAIN):

  files = random.sample(os.listdir(SOURCE), len(os.listdir(SOURCE)))
  for i in range(len(files)):
    source_file = os.path.join(SOURCE, files[i])
    train_file = os.path.join(TRAINING, files[i])
    
    if os.path.getsize(source_file) == 0:
      print(files[i], " is zero length, so ignoring.")
    else:
      if i <= int(len(files)*SPLIT_SIZE_TRAIN):
          copyfile(source_file, train_file)
      else:
          None


folders = os.listdir(root_dir_up)
for i in range(len(folders)):
    source_path_ = os.path.join(root_dir_up,folders[i])
    TRAINING_ = os.path.join(root_dir,folders[i])
    split_data(source_path_, TRAINING_, split_size)


# ##### Number of files in training and testing after random split

os.chdir(root_dir)
folders = os.listdir(root_dir)
for i in range(len(folders)):
    train_path_cat = os.path.join(root_dir,folders[i])
    print(f"{folders[i]}: {len(os.listdir(train_path_cat))} training")


quit()

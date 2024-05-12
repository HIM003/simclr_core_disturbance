import sys
import os
import zipfile
import random
import shutil
from shutil import copyfile


path1 = sys.argv[1] #for_training
path2 = sys.argv[2] #tmp_binary_10train_fold1
split_size_train = float(sys.argv[3]) #0.1
split_size_validation = float(sys.argv[4]) #0.15
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
  os.makedirs(os.path.join(root_dir, "training"))
  os.makedirs(os.path.join(root_dir, "validation"))
  os.makedirs(os.path.join(root_dir, "testing"))
  folders = list(["none", "disturbance"])
  for i in range(len(folders)):
    os.makedirs(os.path.join(root_dir,"training",folders[i]))
    os.makedirs(os.path.join(root_dir,"validation",folders[i]))
    os.makedirs(os.path.join(root_dir,"testing",folders[i]))
 
try:
  create_train_test_dirs(root_dir)
except FileExistsError:
  None 


# ##### Split the images randomly into test and train folders based on split size 
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





folders = os.listdir(root_dir_up)
for i in range(len(folders)):
    source_path_ = os.path.join(root_dir_up,folders[i])
    if folders[i] == "none":
        TESTING_ = os.path.join(root_dir,"testing",folders[i])
        VALIDATION_ = os.path.join(root_dir,"validation",folders[i])
        TRAINING_ = os.path.join(root_dir,"training",folders[i])
    else:
        TESTING_ = os.path.join(root_dir,"testing","disturbance")
        VALIDATION_ = os.path.join(root_dir,"validation","disturbance")
        TRAINING_ = os.path.join(root_dir,"training","disturbance")       
    split_data(source_path_, TRAINING_, VALIDATION_, TESTING_, split_size_train, split_size_validation)


# ##### Number of files in training and testing after random split

os.chdir(root_dir)
folders = os.listdir("training")
for i in range(len(folders)):
    train_path_cat = os.path.join("training",folders[i])
    validation_path_cat = os.path.join("validation",folders[i])
    test_path_cat = os.path.join("testing",folders[i])
    print(f"{folders[i]}: {len(os.listdir(train_path_cat))} training & {len(os.listdir(validation_path_cat))} validation & {len(os.listdir(test_path_cat))} testing ")


quit()

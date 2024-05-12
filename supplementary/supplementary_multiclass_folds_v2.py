import sys
import os
import zipfile
import random
import shutil
from shutil import copyfile


path1 = sys.argv[1] #for_training
path2 = sys.argv[2] #tmp_binary_10train_fold1
split_size_train = float(sys.argv[3]) #0.1
folds = int(sys.argv[4]) #3
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
  for j in range(folds):
    fold_ = "fold"+str(j+1)
    print(fold_)
    os.makedirs(os.path.join(root_dir,fold_,"training"))
    os.makedirs(os.path.join(root_dir,fold_,"validation"))
    os.makedirs(os.path.join(root_dir,fold_,"testing"))
    folders = os.listdir(root_dir_up)
    for i in range(len(folders)):
      os.makedirs(os.path.join(root_dir,fold_,"training",folders[i]))
      os.makedirs(os.path.join(root_dir,fold_,"validation",folders[i]))
      os.makedirs(os.path.join(root_dir,fold_,"testing",folders[i]))
 
try:
  create_train_test_dirs(root_dir)
except FileExistsError:
  None 


# ##### Split the images randomly into test and train folders based on split size 
def split_data(SOURCE, DIST_TYPE, SPLIT_SIZE_TRAIN, FOLDS):

  files = random.sample(os.listdir(SOURCE), len(os.listdir(SOURCE)))
  folders_fold = os.listdir(root_dir)
  #test_files = list() 
  #dict_files = {}

  for i in range(len(files)):
    
    source_file = os.path.join(SOURCE, files[i])
    test_files = list()
    dict_files = {}
    
    for j in range(len(folders_fold)):
      key = "train_file_"+folders_fold[j] 
      print(key)
      val = os.path.join(root_dir, folders_fold[j], "training", DIST_TYPE,files[i])
      dict_files[key] = val 
      key = "validation_file_"+folders_fold[j] 
      val = os.path.join(root_dir, folders_fold[j], "validation", DIST_TYPE,files[i])
      dict_files[key] = val	
      #train_file+"_"+folders_fold[j] = os.path.join(root_dir, folders_fold[j], "training", DIST_TYPE,files[i])
      #validation_file+"_"+folders_fold[j] = os.path.join(root_dir, folders_fold[j], "validation", DIST_TYPE,files[i])
      #test_file+"_"+folders_fold[j] = os.path.join(root_dir, folders_fold[j], "testing", DIST_TYPE, files[i])
      test_files.append(os.path.join(root_dir, folders_fold[j], "testing", DIST_TYPE, files[i]))

    if os.path.getsize(source_file) == 0:
      print(files[i], " is zero length, so ignoring.")
    elif (FOLDS==3):
      if i <= int(len(files)*SPLIT_SIZE_TRAIN):
        copyfile(source_file, dict_files["train_file_fold1"])
        copyfile(source_file, dict_files["validation_file_fold2"])
        copyfile(source_file, dict_files["validation_file_fold3"])
      
      elif  (i > int(len(files)*SPLIT_SIZE_TRAIN)) & (i <= 2*int(len(files)*SPLIT_SIZE_TRAIN)):
        copyfile(source_file, dict_files["train_file_fold2"])
        copyfile(source_file, dict_files["validation_file_fold1"])
        copyfile(source_file, dict_files["validation_file_fold3"])
    
      elif  (i > 2*int(len(files)*SPLIT_SIZE_TRAIN)) & (i <= 3*int(len(files)*SPLIT_SIZE_TRAIN)):
        copyfile(source_file, dict_files["train_file_fold3"])
        copyfile(source_file, dict_files["validation_file_fold1"])
        copyfile(source_file, dict_files["validation_file_fold2"])

      else:
        for k in range(len(test_files)):
          copyfile(source_file, test_files[k])
    else:
      print("Hard-coded number of folds of 3") 

folders = os.listdir(root_dir_up)
for i in range(len(folders)):
  source_path_ = os.path.join(root_dir_up,folders[i])
  dist_type = folders[i] 
  split_data(source_path_, dist_type, split_size_train, folds)


# ##### Number of files in training and testing after random split
os.chdir(root_dir)
folders_fold = os.listdir(root_dir)
for j in range(len(folders_fold)):
  print(folders_fold[j])
  folders = os.listdir(os.path.join(root_dir,folders_fold[j],"training"))
  for i in range(len(folders)):
    train_path_cat = os.path.join(root_dir,folders_fold[j],"training",folders[i])
    validation_path_cat = os.path.join(root_dir,folders_fold[j],"validation",folders[i])
    test_path_cat = os.path.join(root_dir,folders_fold[j],"testing",folders[i])
    print(f"{folders[i]}: {len(os.listdir(train_path_cat))} training & {len(os.listdir(validation_path_cat))} validation & {len(os.listdir(test_path_cat))} testing ")


quit()

import sys
import os
import zipfile
import random
import shutil
from shutil import copyfile


path1 = sys.argv[1] #tmp_383_385_1.8k_binary_fold1
keep = float(sys.argv[2]) #0.095
source_path = '/jmain02/home/J2AD015/axf03/hxm18-axf03/images'
root_dir_up = os.path.join(source_path, path1, "training")
os.chdir(source_path)

def del_data(SOURCE, KEEP):
  files = random.sample(os.listdir(SOURCE), len(os.listdir(SOURCE)))
  print(len(files))
  for i in range(len(files)):
    source_file = os.path.join(SOURCE, files[i])
    if i <= int(len(files)*KEEP):
        None
    else:
        os.remove(source_file)

folders = os.listdir(root_dir_up)
for i in range(len(folders)):
  source_path_ = os.path.join(root_dir_up,folders[i])
  del_data(source_path_, keep)


quit()

import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'data/cat_dog'
Cls_cat = '/cat'
Cls_dog = '/dog'


'''
os.makedirs(root_dir +'/train' + Cls_cat)
os.makedirs(root_dir +'/train' + Cls_dog)
os.makedirs(root_dir +'/val' + Cls_cat)
os.makedirs(root_dir +'/val' + Cls_dog)
os.makedirs(root_dir +'/test' + Cls_cat)
os.makedirs(root_dir +'/test' + Cls_dog)
'''

# Creating partitions of the data after shuffeling
#run for each class
currentCls = Cls_dog
src = "data/cat_dog"+currentCls # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])


train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))

# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, "data/cat_dog/train"+currentCls)

for name in val_FileNames:
    shutil.copy(name, "data/cat_dog/val"+currentCls)

for name in test_FileNames:
    shutil.copy(name, "data/cat_dog/test"+currentCls)

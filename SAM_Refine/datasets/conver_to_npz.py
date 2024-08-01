import os
import numpy as np
import cv2
import nibabel as nib
#from PIL import Image

subfolders = ['1']

# 将fine_data的训练集和验证集转为.npz文件
# Define the base directory

save_dir = '../SAM_FLARE'

ori_dir = 'D:/BaiduNetdiskDownload/FLARE2022/Training/FLARE22_LabeledCase50/images'
label_dir = 'D:/BaiduNetdiskDownload/FLARE2022/Training/FLARE22_LabeledCase50/labels'


a_min, a_max = -125, 275
b_min, b_max = 0.0, 1.0

# Normalize function to scale the data to [0, 1]
def normalize(data):
    # data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data - a_min) / (a_max - a_min)
    return data

# Iterate through the subfolders
for j in range(10,51):
    origin_path = os.path.join(ori_dir, 'FLARE22_Tr_00'+ str(j) + '_0000.nii')
    label_path = os.path.join(label_dir, 'FLARE22_Tr_00' + str(j) + '.nii')

    # List files in the subfolders
    origin_files = os.path.join(origin_path, 'FLARE22_Tr_00'+ str(j) + '_0000.nii')
    label_files = os.path.join(label_path, 'FLARE22_Tr_00' + str(j) + '.nii')

    origin_nib = nib.load(origin_files) # (512,512,112)
    origin_nib = origin_nib.get_fdata()

    label_nib = nib.load(label_files) # (512,512,112)
    label_nib = label_nib.get_fdata()

    # Create .npz files for each pair of corresponding images
    for i in range(origin_nib.shape[2]): # origin_nib AND label_nib

        origin_image = origin_nib[:, :, i]
        label_image = label_nib[:, :, i]

        # ---
        # Set pixels to [0-13]

        origin_image = cv2.normalize(origin_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        origin_image = np.interp(origin_image, (0, 255), (0, 13)).astype(np.uint8)

        label_image = cv2.normalize(label_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        label_image = np.interp(label_image, (0, 255), (0, 13)).astype(np.uint8)
        # ---

        # Load and convert images to numpy arrays

        origin_image = cv2.resize(origin_image,(224,224), fx=0.64, fy=0.64, interpolation=cv2.INTER_LINEAR)
        #origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)


        label_image = cv2.resize(label_image,(224,224), fx=0.64, fy=0.64, interpolation=cv2.INTER_LINEAR)
        #label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)

        origin_image = origin_image.astype('float32')
        #label_image = label_image.astype('int64')
        label_image = label_image.astype('float32')

        #
        origin_image = np.clip(origin_image, a_min, a_max)

        origin_image = normalize(origin_image)

        # Define the file name for the .npz file
        case_name = f'case{int(j):04d}_slice{int(i):04d}.npz'
        npz_file_path = os.path.join(save_dir, case_name)

        # Save the images as 'image' and 'label' in the .npz file
        np.savez(npz_file_path, image=origin_image, label=label_image)
        print(f'Saved {npz_file_path}')

import os
import numpy as np
import cv2
import nibabel as nib
import h5py

subfolders = ['1']

# 将fine_data的训练集和验证集转为.npz文件
# Define the base directory

output_h5_file_path = "../testset/test1000"

ori_dir = 'D:/BaiduNetdiskDownload/FLARE2022/Training/FLARE22_UnlabeledCase1-1000'
label_dir = 'D:/BaiduNetdiskDownload/FLARE2022/Training/FLARE22_UnlabeledCase1-1000'


a_min, a_max = -125, 275
b_min, b_max = 0.0, 1.0

# Normalize function to scale the data to [0, 1]
def normalize(data):
    # data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data - a_min) / (a_max - a_min)
    return data

for i in range(100,1000):

    origin_files = os.path.join(ori_dir, 'Case_' + '00' + str(i) + '_0000.nii.gz')
    label_files = os.path.join(ori_dir, 'Case_' + '00' + str(i) + '_0000.nii.gz')
    output_h5_file = os.path.join(output_h5_file_path, 'case00' + str(i) + '.npy.h5')

    origin_nib = nib.load(origin_files)  # (512,512,112)
    origin_nib = origin_nib.get_fdata()

    label_nib = nib.load(label_files)  # (512,512,112)
    label_nib = label_nib.get_fdata()

    z_dim = origin_nib.shape[2]

    with h5py.File(output_h5_file, "w") as hdf5_file:
        # Create 'image' and 'label' datasets
        image_data = np.zeros((z_dim, 512, 512), dtype=np.float32)
        label_data = np.zeros((z_dim, 512, 512), dtype=np.float32)

        for j in range(z_dim):
            origin_image = origin_nib[:, :, j]
            label_image = label_nib[:, :, j]
            # ---
            # Set pixels to [0-13]

            origin_image = cv2.normalize(origin_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            origin_image = np.interp(origin_image, (0, 255), (0, 13)).astype(np.uint8)

            label_image = cv2.normalize(label_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            label_image = np.interp(label_image, (0, 255), (0, 13)).astype(np.uint8)
            # ---

            # Load and convert images to numpy arrays

            origin_image = cv2.resize(origin_image, (512, 512), fx=0.64, fy=0.64, interpolation=cv2.INTER_LINEAR)
            # origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

            label_image = cv2.resize(label_image, (512, 512), fx=0.64, fy=0.64, interpolation=cv2.INTER_LINEAR)
            # label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)

            origin_image = origin_image.astype('float32')
            label_image = label_image.astype('int64')

            #
            origin_image = np.clip(origin_image, a_min, a_max)

            origin_image = normalize(origin_image)

            # # Store in the datasets
            image_data[j] = origin_image
            label_data[j] = label_image

            # Create datasets for 'image' and 'label'
        hdf5_file.create_dataset("image", data=image_data)
        hdf5_file.create_dataset("label", data=label_data)

    print(
        f"HDF5 file '{output_h5_file}' has been created with 'image' and 'label' datasets, each with a shape (28, 512, 512) and data type '<f4'.")
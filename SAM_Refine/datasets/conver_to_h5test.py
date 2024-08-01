import h5py
import os
import cv2
import numpy as np
import nibabel as nib

data_folder_path = 'D:/BaiduNetdiskDownload/FLARE2022/Training/FLARE22_UnlabeledCase1-1000'
output_h5_file_path = "../testset/test1000"

# Normalize function to scale the data to [0, 1]

a_min, a_max = -125, 275
b_min, b_max = 0.0, 1.0

# Normalize function to scale the data to [0, 1]
def normalize(data):
    # data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data - a_min) / (a_max - a_min)
    return data

for i in range(1,10):

    # Case_00001_0000.nii.gz
    data_folder = os.path.join(data_folder_path,'Case_' + '0000' + str(i) + '_0000.nii.gz')
    image = nib.load(data_folder)
    data = image.get_fdata()
    z_dim = data.shape[2]

    #
    output_h5_file = os.path.join(output_h5_file_path,'case0000'+ str(i) + '.npy.h5')

    # Create an HDF5 file for writing
    with h5py.File(output_h5_file, "w") as hdf5_file:
        # Create 'image' and 'label' datasets
        image_data = np.zeros((z_dim, 512, 512), dtype=np.float32)
        label_data = np.zeros((z_dim, 512, 512), dtype=np.float32)

        for j in range(z_dim):

            origin_image = data[:, :, j]
            label_image = data[:, :, j]

            origin_image = origin_image.astype('float32')
            label_image = label_image.astype('int64')

            origin_image = np.clip(origin_image, a_min, a_max)

            # Normalize the images to (0, 1)
            origin_image = normalize(origin_image)

            # # Store in the datasets
            image_data[j] = origin_image
            label_data[j] = label_image

        # Create datasets for 'image' and 'label'
        hdf5_file.create_dataset("image", data=image_data)
        hdf5_file.create_dataset("label", data=label_data)


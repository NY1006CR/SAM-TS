# Teacher
# 原始图像->初始伪标签
import torch
import numpy as np
import onnxruntime
import cv2
import os
import random
from paddle.vision.transforms import hflip, vflip, ColorJitter
from paddle.vision.transforms.functional import rotate

#---设置一些文件路径
save_path = '../SAM_Refine/dataset_pseudo' # 伪标签保存路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

image_path = "data/test" # 原始图像路径
if not os.path.exists(image_path):
    os.makedirs(image_path)

onnx_path = '../SAM_Refine/checkpoint' # onnx权重路径
if not os.path.exists(onnx_path):
    os.makedirs(onnx_path)

times = 4 # 定义几次选择
# 图像列表.txt
val_dataset = []

#---

# TODO 读取image_path下所有图片
for i in range(len(val_dataset)):
    # 根据数据集结构名称进行调整
    val_image = cv2.imread(os.path.join(image_path,'0' + str(i+100) + '.png'), cv2.IMREAD_GRAYSCALE)
    images = []
    for i in range(times):
        # onnx模型只接受（512*512）
        val_image = cv2.resize(val_image, (512, 512), interpolation=cv2.INTER_LINEAR)
        val_image = np.array(val_image)
        val_image = np.expand_dims(val_image, axis=2)
        h, w, _ = val_image.shape
        val_image = val_image.astype('float32')
        val_image = cv2.cvtColor(val_image, cv2.COLOR_GRAY2BGR)

        imgA = val_image
        # TODO 进行图像增强 并记录使用了何种方法
        enhance_list = []

        if random.random() > 0.5:
            # 随机旋转
            angle = random.randint(0, 60)
            imgA = rotate(imgA, angle)

            enhance_list.append(1)
        # 随机水平翻转 和垂直翻转
        if random.random() > 0.5:
            imgA = hflip(imgA)

            enhance_list.append(2)
        if random.random() > 0.5:
            imgA = vflip(imgA)
            enhance_list.append(3)
        if random.random() > 0.5:
            # 随机调整图像的亮度，对比度，饱和度和色调
            val = round(random.random() / 3, 1)
            color = ColorJitter(val, val, val, val)
            imgA = imgA.astype('uint8')
            imgA = color(imgA)
            imgA = imgA.astype('float32')
            enhance_list.append(4)
        if random.random() > 0.2:
            # 随机生成4个小黑色方块遮挡
            for i in range(4):
                black_width = 50
                black_height = 50
                width, height, _ = imgA.shape
                loc1 = random.randint(0, (width - black_width - 1))
                loc2 = random.randint(0, (height - black_width - 1))
                imgA[loc1:loc1 + black_width, loc2:loc2 + black_height:, :] = 0
            enhance_list.append(5)

        val_image = val_image / 255.
        val_image = np.transpose(val_image, (2, 0, 1))

        imga = val_image

        imga = np.expand_dims(imga, axis=0)

        segmentation = np.zeros((512,512,1))
        x_data = torch.tensor(imga, dtype=torch.float32)

        #
        model = onnxruntime.InferenceSession(onnx_path) # bone

        ort_inputs = {model.get_inputs()[0].name: x_data.numpy()} # x_data是个tensor 需要将其转为numpy
        output = model.run(None, ort_inputs) # class list
        output = output[0]
        output = np.argmax(output,axis=1)
        output = output.transpose(1,2,0)
        #
        # # 根据需求更改特征的像素值
        # for i in np.arange(512):
        #     for j in np.arange(512):
        #         if output[i,j,0] == 2:
        #             output[i,j,0] = 50
        #         if output[i,j,0] == 1:
        #             output[i,j,0] = 100


        # TODO 根据之前图像增强对预测结果进行还原

        # TODO 保存伪标签 用[]保存
        images.append(output)
        # cv2.imwrite(os.path.join(save_path, 'pseudo' + str(i) + '.jpg'), output)

    # TODO 对保存的图像数组进行判断 IOU

    # TODO if state=0 else state=1

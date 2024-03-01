###\u6709\u5750\u6807\u56fe\u8f93\u5165\u7684\u5355\u6a21\u578b\u591a\u56fe\u8bc4\u4ef7\u65b9\u6cd5
import argparse
import cv2
import math
import numpy as np
import os
import torch
import time
from torch.autograd import Variable
from libtiff import TIFF, TIFFfile, TIFFimage
from networks.DPDN import DPDN
from networks.DPG import DPG
from PIL import Image
from scipy import signal
from My_function import reorder_imec, reorder_2filter


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return signal.convolve2d(x, np.rot90(kernel, 2), mode=mode)

def mask_input(GT_image):
    # 创建一个全零矩阵，与输入图像大小相同，用于存储马赛克滤波后的图像
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], 16), dtype=np.float32)
    
    # 根据特定的滤波模式，将某些位置设置为1，形成16个子图像块
    mask[0::4, 0::4, 0] = 1
    mask[0::4, 1::4, 1] = 1
    mask[0::4, 2::4, 2] = 1
    mask[0::4, 3::4, 3] = 1
    mask[1::4, 0::4, 4] = 1
    mask[1::4, 1::4, 5] = 1
    mask[1::4, 2::4, 6] = 1
    mask[1::4, 3::4, 7] = 1
    mask[2::4, 0::4, 8] = 1
    mask[2::4, 1::4, 9] = 1
    mask[2::4, 2::4, 10] = 1
    mask[2::4, 3::4, 11] = 1
    mask[3::4, 0::4, 12] = 1
    mask[3::4, 1::4, 13] = 1
    mask[3::4, 2::4, 14] = 1
    mask[3::4, 3::4, 15] = 1
    
    # 将输入图像与马赛克滤波后的图像进行元素级乘法
    input_image = mask * GT_image
    
    # 返回经过马赛克滤波处理后的图像
    return input_image


def compute_PSNR(estimated,real):
    estimated = np.float64(estimated)
    real = np.float64(real)
    MSE = np.mean((estimated-real)**2)
    PSNR = 10*np.log10(255*255/MSE)
    return PSNR

def compute_ssim_channel(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))

def compute_ssim(estimate,real):
    SSIM_totoal  =0
    for i in range(estimate.shape[2]):
        SSIM = compute_ssim_channel(estimate[:, :, i], real[:, :, i])
        SSIM_totoal += SSIM
    return SSIM_totoal / 16.0

def compute_sam(x_true, x_pre):
    """
    计算光谱相角（Spectral Angle Mapper，SAM）。

    SAM 是一种用于比较两个光谱向量之间的相似性的指标，通常用于遥感图像分析和匹配。

    参数:
    x_true (numpy.ndarray): 真实的光谱向量。
    x_pre (numpy.ndarray): 预测的光谱向量。

    返回:
    float: 以度为单位的光谱相角。

    计算步骤:
    1. 计算真实光谱向量和预测光谱向量按元素相乘的结果。
    2. 沿光谱波段维度对上一步的结果进行求和，得到两个向量的点积。
    3. 避免除零错误，将点积中的零元素替换为接近零的值。
    4. 计算真实光谱向量的 L2 范数，用于归一化分母。
    5. 避免除零错误，将 L2 范数中的零元素替换为接近零的值。
    6. 计算预测光谱向量的 L2 范数，用于归一化分母。
    7. 避免除零错误，将 L2 范数中的零元素替换为接近零的值。
    8. 计算两个向量的余弦相似性，即点积除以两个范数的乘积。
    9. 归一化余弦相似性，确保其值在0到1之间。
    10. 截断大于1的值为1，以确保余弦相似性不会超出1。
    11. 计算余弦相似性的反余弦值，即光谱相角（SAM）的弧度表示。
    12. 计算整个图像的平均光谱相角。
    13. 将平均相角转换为角度制，并以度为单位返回 SAM。

    SAM 通常用于遥感图像中不同像素点之间的光谱相似性度量，可用于分类、目标检测和光谱匹配等应用中。
    """
    # 以下为函数实现
    
    buff1 = x_true * x_pre
    buff2 = np.sum(buff1, 2)
    buff2[buff2 == 0] = 2.2204e-16
    buff4 = np.sqrt(np.sum(x_true * x_true, 2))
    buff4[buff4 == 0] = 2.2204e-16
    buff5 = np.sqrt(np.sum(x_pre * x_pre, 2))
    buff5[buff5 == 0] = 2.2204e-16
    buff6 = buff2 / buff4
    buff8 = buff6 / buff5
    buff8[buff8 > 1] = 1
    buff10 = np.arccos(buff8)
    buff9 = np.mean(np.arccos(buff8))
    SAM = (buff9) * 180 / np.pi
    return SAM


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
# parser.add_argument("--model", default="/home/dell/wyz/workGJS/DDM-Net/checkpoint/model_in_paper.pth", type=str, help="model path")
parser.add_argument("--model", default="/home/dell/wyz/workGJS/DDM-Net/checkpoint/pre-training/PPI_model_epoch_2000_in_paper.pth", type=str, help="model path")
#parser.add_argument("--model", default="checkpoint/fine-tuning/final_model_epoch_4000.pth", type=str, help="model path")
parser.add_argument("--dataset", default="/home/dell/wyz/workGJS/dataset/RealIMG_GJS/test", type=str, help="dataset name, Default: CAVE")
parser.add_argument("--scale", default=4, type=int, help="msfa_size, Default: 4")

opt = parser.parse_args()
cuda = True
norm_flag = False

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print(opt.model)

# model= DPDN()
PPI_net = DPG()

m_state_dict = torch.load(opt.model)
# model.load_state_dict(m_state_dict['model'].state_dict())
PPI_net.load_state_dict(m_state_dict['model'].state_dict())

image_list = opt.dataset
avg_psnr_predicted = 0.0
avg_psnr_PPID = 0.0
avg_sam_predicted = 0.0
avg_sam_PPID = 0.0
avg_ssim_predicted_MCAN = 0.0
avg_ssim_predicted_our= 0.0
avg_ssim_PPID = 0.0
avg_ergas_predicted = 0.0
avg_elapsed_time = 0.0
avg_PPI_psnr_predicted = 0.0
sample_num = 0
knum = 1
with torch.no_grad():
    for ijk in range(knum):
        for i in range(1):
            for image_name in sorted(os.listdir(image_list)):
                print("Processing", image_name)
                sample_num = sample_num + 1
                im_gt_y = np.load(opt.dataset + "/" + image_name)  # 512*512*16

                raw = im_gt_y

                raw = Variable(torch.from_numpy(raw).float()).view(1, -1, raw.shape[0], raw.shape[1])
                
                if cuda:
                    PPI_net = PPI_net.cuda()
                    # model = model.cuda()
                    raw = raw.cuda()
                else:
                    PPI_net  =  PPI_net.cpu()
                    # model = model.cpu()

                estimated_PPI = PPI_net(raw)
                # 使用深度学习模型进行推断，将稀疏图像(sparse_image)和原始数据(raw)输入到模型中

                # 将estimated_PPI从GPU移到CPU上，以便后续处理
                estimated_PPI = estimated_PPI.cpu()
                # 将estimated_PPI的数据从PyTorch张量转换为NumPy数组，并将数据类型更改为float32
                estimated_PPI = estimated_PPI.data[0].numpy().astype(np.float32)
                
                # 将estimated_PPI的像素值缩放到0-255范围内
                estimated_PPI = estimated_PPI * 255.
                estimated_PPI[estimated_PPI < 0] = 0
                estimated_PPI[estimated_PPI > 255.] = 255.

                # 将raw从GPU移到CPU上
                raw = raw.cpu()
                # 将raw的数据从PyTorch张量转换为NumPy数组，并将数据类型更改为float32
                raw = raw.data[0].numpy().astype(np.float32)
                # 将raw的像素值缩放到0-255范围内
                raw = raw * 255.
                raw[raw < 0] = 0
                raw[raw > 255.] = 255.

                # 将im_gt_y的像素值缩放到0-255范围内
                im_gt_y = im_gt_y * 255.
                im_gt_y[im_gt_y < 0] = 0
                im_gt_y[im_gt_y > 255.] = 255.
                
                # 创建目录名和文件名，将结果保存为图像文件
                kind = image_name[:-4]
                kind_dir = os.path.join('test_demosaic_result/RealIMG_GJS/' + kind + '/')
                os.makedirs(kind_dir, exist_ok=True)
                PPI_path = os.path.join(kind_dir + '/estimated_PPI.png')
                cv2.imwrite(PPI_path,estimated_PPI[0,:,:])

                demosaic_real_path = os.path.join(kind_dir + '/raw.png')
                cv2.imwrite(demosaic_real_path, raw[0,:,:])

                del estimated_PPI
                del raw
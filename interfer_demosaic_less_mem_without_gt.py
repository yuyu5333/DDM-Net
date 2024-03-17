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
from PIL import Image
from scipy import signal
from My_function import reorder_imec, reorder_2filter, reorder_imec_GJS

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
    
    np.save('/home/dell/wyz/workGJS/DDM-Net/test_demosaic_result/SaveTempValue/input_image_mask.npy', input_image)
    print("save input_image_mask success")
    
    # 返回经过马赛克滤波处理后的图像
    return input_image

def mask_input_25(GT_image):
    # 创建一个全零矩阵，与输入图像大小相同，用于存储马赛克滤波后的图像
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], 25), dtype=np.float32)
    
    # 根据特定的滤波模式，将某些位置设置为1，形成25个子图像块
    mask[0::5, 0::5, 0] = 1
    mask[0::5, 1::5, 1] = 1
    mask[0::5, 2::5, 2] = 1
    mask[0::5, 3::5, 3] = 1
    mask[0::5, 4::5, 4] = 1
    
    mask[1::5, 0::5, 5] = 1
    mask[1::5, 1::5, 6] = 1
    mask[1::5, 2::5, 7] = 1
    mask[1::5, 3::5, 8] = 1
    mask[1::5, 4::5, 9] = 1
    
    mask[2::5, 0::5, 10] = 1
    mask[2::5, 1::5, 11] = 1
    mask[2::5, 2::5, 12] = 1
    mask[2::5, 3::5, 13] = 1
    mask[2::5, 4::5, 14] = 1
    
    mask[3::5, 0::5, 15] = 1
    mask[3::5, 1::5, 16] = 1
    mask[3::5, 2::5, 17] = 1
    mask[3::5, 3::5, 18] = 1
    mask[3::5, 4::5, 19] = 1
    
    mask[4::5, 0::5, 20] = 1
    mask[4::5, 1::5, 21] = 1
    mask[4::5, 2::5, 22] = 1
    mask[4::5, 3::5, 23] = 1
    mask[4::5, 4::5, 24] = 1
    
    # 将输入图像与马赛克滤波后的图像进行元素级乘法
    input_image = mask * GT_image
    
    np.save('/home/dell/wyz/workGJS/DDM-Net/test_demosaic_result/SaveTempValue/input_image_mask.npy', input_image)
    print("save input_image_mask success")
    
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


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="/home/dell/wyz/workGJS/DDM-Net/checkpoint/main/main_model_epoch_500.pth", type=str, help="model path")
#parser.add_argument("--model", default="checkpoint/fine-tuning/final_model_epoch_4000.pth", type=str, help="model path")
# parser.add_argument("--dataset", default="/home/dell/wyz/workGJS/dataset/RealIMG_GJS/test_repeat16", type=str, help="dataset name, Default: CAVE")
parser.add_argument("--dataset", default="/home/dell/wyz/workGJS/dataset/MascDataSetMyMade/interfer1time", type=str, help="dataset name, Default: CAVE")
# parser.add_argument("--dataset", default="/home/dell/wyz/workGJS/dataset/RealIMG_GJS/scenery_npy_25_dim25", type=str, help="dataset name, Default: CAVE")

parser.add_argument("--dataone", default="/home/dell/wyz/workGJS/dataset/MascDataSetMyMade/TrainALL/shop1.npy", type=str, help="dataset name, Default: CAVE")
parser.add_argument("--scale", default=5, type=int, help="msfa_size, Default: 4")

# parser.add_argument("--result_dir", default="RealIMG_GJS/scenery/")
parser.add_argument("--result_dir", default="Temp/")

opt = parser.parse_args()
cuda = True
norm_flag = False

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print(opt.model)

model= DPDN()
m_state_dict = torch.load(opt.model)
model.load_state_dict(m_state_dict['model'].state_dict())

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
                # im_gt_y = np.load(opt.dataset + "/" + image_name)  # 512*512*16
                # # 使用实际相机滤波阵列对原始的im_gt_y进行马赛克滤波列排列
                # im_l_y = mask_input(im_gt_y)
                # # 按照实际相机滤波阵列的顺序排列列逆还原为从大到小的顺序
                # im_l_y = reorder_imec(im_l_y)
                # im_gt_y = reorder_imec(im_gt_y)
                # im_input = im_l_y

                # im_input 就是马赛克图像，加载一个马赛克图像进来
                # im_input = np.load(opt.dataset + "/" + image_name)  # 512*512*16
                
                # im_l_y 是拥有16个维度的图像，每个维度是稀疏的图像，加载一个16维度的马赛克图像，再处理一下每个维度的稀疏性
                # input_raw_16 = np.load(opt.dataset + "/" + image_name)
                
                # 使用实际相机滤波阵列对原始的input_raw_16进行马赛克滤波列排列
                # im_l_y = mask_input(input_raw_16)
                
                im_gt_y = np.load(opt.dataone)
                
                # im_gt_y = np.load(opt.dataset + "/" + image_name)
                
                # # 使用实际相机滤波阵列对原始的im_gt_y进行马赛克滤波列排列
                im_l_y = mask_input_25(im_gt_y)
                # 按照实际相机滤波阵列的顺序排列列逆还原为从大到小的顺序
                im_l_y = reorder_imec_GJS(im_l_y)

                im_input = im_l_y

                # 将通道维度移动到最前面，以适应PyTorch模型输入要求
                # im_gt_y = im_gt_y.transpose(2, 0, 1)
                im_l_y = im_l_y.transpose(2, 0, 1)
                im_input = im_input.transpose(2, 0, 1) # C H W
                raw = im_input.sum(axis=0)   # MSFA

                # print("here")

                im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[1], im_input.shape[2])
                raw = Variable(torch.from_numpy(raw).float()).view(1, -1, raw.shape[0], raw.shape[1])
                estimated_Demosaic = np.zeros_like(im_l_y)
                data_mat_sparse_image = {}
                # ToDo 确认波段
                for index in range((opt.scale) ** 2):
                    # 从im_l_y中选择特定通道的子图像
                    sparse_image = im_l_y[index, :, :]
                    data_mat_sparse_image[index] = sparse_image
                    # 将sparse_image转换为PyTorch张量，并进行形状变换以满足深度学习模型的输入要求
                    sparse_image = Variable(torch.from_numpy(sparse_image).float()).view(1, -1, sparse_image.shape[0], sparse_image.shape[1])
                    if cuda:
                        # PPI_net = PPI_net.cuda()
                        model = model.cuda()

                        im_input = im_input.cuda()
                        raw = raw.cuda()
                        sparse_image = sparse_image.cuda()
                    else:
                        # PPI_net  =  PPI_net.cpu()
                        model = model.cpu()

                    # estimated_PPI = PPI_net(raw)
                    # 使用深度学习模型进行推断，将稀疏图像(sparse_image)和原始数据(raw)输入到模型中
                    
                    estimated_PPI, estimated_demosaic = model(raw, sparse_image)

                    # 将estimated_demosaic从GPU移到CPU上，以便后续处理
                    estimated_demosaic = estimated_demosaic.cpu()

                    # 将estimated_demosaic的数据从PyTorch张量转换为NumPy数组，并将数据类型更改为float32
                    estimated_demosaic = estimated_demosaic.data[0].numpy().astype(np.float32)

                    # 将estimated_demosaic的像素值缩放到0-255范围内
                    estimated_demosaic = estimated_demosaic * 255.0

                    # 将小于0的像素值截断为0，将大于255的像素值截断为255，确保像素值在有效范围内
                    estimated_demosaic[estimated_demosaic < 0] = 0
                    estimated_demosaic[estimated_demosaic > 255] = 255

                    # 将估计的Demosaic结果(estimated_demosaic)存储在结果数组(estimated_Demosaic)中的特定通道(index)中
                    estimated_Demosaic[index, :, :] = estimated_demosaic

                # 计算图像im_gt_y每个通道的像素和并除以16，得到平均值
                # target_PPI = im_gt_y.sum(axis=0) / 16.
                # 将target_PPI转换为PyTorch的Variable，并将数据类型转换为float32
                # target_PPI = Variable(torch.from_numpy(target_PPI).float()).view(1, -1, target_PPI.shape[0], target_PPI.shape[1])

                # 将target_PPI从CPU移到GPU上（如果CUDA可用）
                # target_PPI = target_PPI.cuda()
                # target_PPI = target_PPI

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

                # 将im_input从GPU移到CPU上
                im_input = im_input.cpu()
                im_input = im_input.data[0].numpy().astype(np.float32)
                im_input = im_input * 255.
                im_input[im_input < 0] = 0
                im_input[im_input > 255.] = 255.

                # 将target_PPI从GPU移到CPU上
                # target_PPI = target_PPI.cpu()
                # target_PPI = target_PPI.data[0].numpy().astype(np.float32)
                # target_PPI = target_PPI * 255.
                # target_PPI[target_PPI < 0] = 0
                # target_PPI[target_PPI > 255.] = 255.

                # 将im_gt_y的像素值缩放到0-255范围内
                # im_gt_y = im_gt_y * 255.
                # im_gt_y[im_gt_y < 0] = 0
                # im_gt_y[im_gt_y > 255.] = 255.

                # 计算估计PPI与真实PPI之间的PSNR
                # PPI_psnr_predicted = compute_PSNR(estimated_PPI[0,:,:],target_PPI[0,:,:])
                # print("PSNR_PPI      =",PPI_psnr_predicted)
                
                # psnr_predicted = compute_PSNR(im_gt_y.transpose(2, 1, 0),estimated_Demosaic.transpose(2, 1, 0))
                # print("PSNR_Demosaic =", psnr_predicted)
                
                # ssim_predicted_our = compute_ssim(im_gt_y.transpose(2, 1, 0), estimated_Demosaic.transpose(2, 1, 0))
                # print("SSIM_Demosaic =", ssim_predicted_our)
                
                # sam_predicted = compute_sam(im_gt_y.transpose(2, 1, 0), estimated_Demosaic.transpose(2, 1, 0))
                # print("SAM_Demosaic  =", sam_predicted)
                
                # 创建目录名和文件名，将结果保存为图像文件
                kind = image_name[:-4]
                result_dir = '/home/dell/wyz/workGJS/DDM-Net/test_demosaic_result/'
                kind_dir = os.path.join(result_dir + opt.result_dir + kind + '/')
                print(kind_dir)
                os.makedirs(kind_dir, exist_ok=True)
                PPI_path = os.path.join(kind_dir + '/estimated_PPI.png')
                cv2.imwrite(PPI_path,estimated_PPI[0,:,:])
                # PPI_real_path = os.path.join(kind_dir + '/real_PPI.png')
                # cv2.imwrite(PPI_real_path, target_PPI[0,:,:])
                
                # 循环保存估计的Demosaic通道和真实的Demosaic通道作为图像文件
                for channel in range((opt.scale) ** 2):
                    demosaic_path = os.path.join(kind_dir + '/estimated_channel_'+str(channel)+'.png')
                    cv2.imwrite(demosaic_path, estimated_Demosaic[channel, :, :])

                    # demosaic_real_path = os.path.join(kind_dir + '/real_channel_' + str(channel) + '.png')
                    # cv2.imwrite(demosaic_real_path, im_gt_y[channel, :, :])

                demosaic_real_path = os.path.join(kind_dir +  '/raw.png')
                cv2.imwrite(demosaic_real_path, raw[0,:,:])

                # 更新PSNR和SAM的平均值
                # avg_PPI_psnr_predicted += PPI_psnr_predicted
                # avg_psnr_predicted += psnr_predicted
                # avg_sam_predicted += sam_predicted
                # avg_ssim_predicted_our += ssim_predicted_our

                del estimated_PPI
                del raw
                del im_input

print("Dataset   :", opt.dataset)
# print('sample_num:',sample_num)
# print("PPI_PSNR_avg_predicted=", avg_PPI_psnr_predicted/sample_num)
# print("PSNR    _avg_predicted=", avg_psnr_predicted / sample_num)
# print("SSIM    _avg_predicted=", avg_ssim_predicted_our / sample_num)
# print("SAM     _avg_predicted=", avg_sam_predicted / sample_num)

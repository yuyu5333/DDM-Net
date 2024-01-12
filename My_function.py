import numpy as np
import torch
from libtiff import TIFFfile

def msfaTOcube(raw, msfa_size):
    mask = np.zeros((raw.shape[0], raw.shape[1], msfa_size**2), dtype=np.int)
    cube = np.zeros((raw.shape[0], raw.shape[1], msfa_size**2), dtype=np.int)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[i::msfa_size, j::msfa_size, i * msfa_size + j] = 1
    for i in range(msfa_size**2):
        cube[:, :, i] = raw * (mask[:, :, i])
    return cube

def mask_input(GT_image, msfa_size):
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], msfa_size ** 2), dtype=np.float32)
    for i in range(0,msfa_size):
        for j in range(0,msfa_size):
            mask[i::msfa_size, j::msfa_size, i*msfa_size+j] = 1
    input_image = mask * GT_image
    return input_image

def sparse_input_channel(GT_image, msfa_size,channel):
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], msfa_size ** 2), dtype=np.float32)
    for i in range(0, msfa_size):
        for j in range(0, msfa_size):
            mask[i::msfa_size, j::msfa_size, i * msfa_size + j] = 1
    input_image = mask * GT_image
    return input_image[:,:,channel]

def reorder_imec(old):
    """
    重新排列多频带立体影像数据，使中心波长从小到大排列。

    参数:
    old (numpy.ndarray): 输入的多频带立体影像数据，具有形状 (_, _, C)，其中 C 是通道数。

    返回:
    numpy.ndarray: 重新排列后的多频带立体影像数据，具有相同的形状 (_, _, C)。
    """
    ### reorder the multiband cube, making the center wavelength from small to large
    
    _, _, C = old.shape  # 获取输入数据的形状

    new = np.zeros_like(old)  # 创建一个与输入数据相同形状的新数组

    if C == 16:
        # new[:, :, 0] = old[:, :, 2]
        # new[:, :, 1] = old[:, :, 0]
        # new[:, :, 2] = old[:, :, 9]
        # new[:, :, 3] = old[:, :, 1]
        # new[:, :, 4] = old[:, :, 15]
        # new[:, :, 5] = old[:, :, 14]
        # new[:, :, 6] = old[:, :, 12]
        # new[:, :, 7] = old[:, :, 13]
        # new[:, :, 8] = old[:, :, 7]
        # new[:, :, 9] = old[:, :, 6]
        # new[:, :, 10] = old[:, :, 4]
        # new[:, :, 11] = old[:, :, 5]
        # new[:, :, 12] = old[:, :, 11]
        # new[:, :, 13] = old[:, :, 10]
        # new[:, :, 14] = old[:, :, 8]
        # new[:, :, 15] = old[:, :, 3]
        new[:, :, 0] = old[:, :, 6]
        new[:, :, 1] = old[:, :, 7]
        new[:, :, 2] = old[:, :, 5]
        new[:, :, 3] = old[:, :, 4]
        new[:, :, 4] = old[:, :, 14]
        new[:, :, 5] = old[:, :, 15]
        new[:, :, 6] = old[:, :, 13]
        new[:, :, 7] = old[:, :, 12]
        new[:, :, 8] = old[:, :, 10]
        new[:, :, 9] = old[:, :, 11]
        new[:, :, 10] = old[:, :, 9]
        new[:, :, 11] = old[:, :, 8]
        new[:, :, 12] = old[:, :, 2]
        new[:, :, 13] = old[:, :, 3]
        new[:, :, 14] = old[:, :, 1]
        new[:, :, 15] = old[:, :, 0]
        return new
    elif C==25:
        new[:, :, 0] = old[:, :, 2]
        new[:, :, 1] = old[:, :, 9]
        new[:, :, 2] = old[:, :, 14]
        new[:, :, 3] = old[:, :, 13]
        new[:, :, 4] = old[:, :, 12]
        new[:, :, 5] = old[:, :, 10]
        new[:, :, 6] = old[:, :, 11]
        new[:, :, 7] = old[:, :, 8]
        new[:, :, 8] = old[:, :, 7]
        new[:, :, 9] = old[:, :, 5]
        new[:, :, 10] = old[:, :, 6]
        new[:, :, 11] = old[:, :, 23]
        new[:, :, 12] = old[:, :, 22]
        new[:, :, 13] = old[:, :, 20]
        new[:, :, 14] = old[:, :, 21]
        new[:, :, 15] = old[:, :, 3]
        new[:, :, 16] = old[:, :, 0]
        new[:, :, 17] = old[:, :, 1]
        new[:, :, 18] = old[:, :, 18]
        new[:, :, 19] = old[:, :, 17]
        new[:, :, 20] = old[:, :, 15]
        new[:, :, 21] = old[:, :, 16]
        new[:, :, 22] = old[:, :, 24]
        new[:, :, 23] = old[:, :, 19]
        new[:, :, 24] = old[:, :, 4]
        return new

def reorder_2filter(old):
    ###从波段中心从小到大的排列，reorder为滤波器从左往右依次的顺序（也就是GT图中的顺序）
    ### reorder the multiband cube as the real pattern in MSFA
    C,_,_ = old.shape
    new = np.zeros_like(old)
    if C == 16:
        # order = [2, 0, 9, 1, 15, 14, 12, 13, 7, 6, 4, 5, 11, 10, 8, 3]
        order = [6, 7, 5, 4, 14, 15, 13, 12, 10, 11, 9, 8, 2, 3, 1, 0]
        for i in range(0, 16):
            new[order[i], :, :] = old[i, :, :]
        return new
    elif C==25:
        order =[2, 9, 14, 13, 12, 10, 11, 8, 7, 5, 6, 23, 22, 20, 21, 3, 0, 1, 18, 17, 15, 16, 24, 19, 4]
        for i in range(0, 25):
            new[order[i], :, :] = old[i, :, :]
        return new

def input_matrix_wpn(inH, inW, msfa_size):
    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    for i in range(0,msfa_size):
        h_offset_coord[i::msfa_size, :, 0] = (i+1)/msfa_size
        w_offset_coord[:, i::msfa_size, 0] = (i+1)/msfa_size
    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1,2)
    return pos_mat

def load_img(filepath):
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0].transpose(2, 1, 0)
    return img

def normalization(x):
    """"
    归一化到区间{0,1]
    返回副本
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range

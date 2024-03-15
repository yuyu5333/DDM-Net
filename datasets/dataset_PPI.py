from os import listdir
from os.path import join

import torch.utils.data as data
from libtiff import TIFFfile
from PIL import Image
import  numpy as np
import random
from My_function import reorder_imec, mask_input

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])

def load_img(filepath):
    # img = Image.open(filepath+'/1.tif')
    # y = np.array(img).reshape(1,img.size[0],img.size[1])
    # m = np.tile(y, (2, 1, 1))
    # tif = TIFFfile(filepath+'/IMECMine_D65.tif')
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0].transpose(2, 1, 0)
    # img_test = Image.fromarray(img[:,:,1])
    return img

def randcrop(a, crop_size):
    [wid, hei, nband]=a.shape
    crop_size1 = crop_size
    Width = random.randint(0, wid - crop_size1 - 1)
    Height = random.randint(0, hei - crop_size1 - 1)

    return a[Width:(Width + crop_size1),  Height:(Height + crop_size1), :]

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

band_msc = 25
band_msc_f = 25.0
mask = 5

class DatasetFromFolder_PPI(data.Dataset):
    def __init__(self, image_dir,norm_flag, input_transform=None, target_transform=None, augment=False):
        super(DatasetFromFolder_PPI, self).__init__()
        # print(listdir(image_dir))
        # for y in listdir(image_dir):
        #     print(y)
        #     if is_image_file(y):
        #         print(y)
        #print(join(image_dir, x))
        # self.image_filenames = [join(image_dir, x, x) for x in listdir(image_dir)]
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir)]
        print(self.image_filenames)
        random.shuffle(self.image_filenames)
        print(self.image_filenames)
        #ToDo 确认这里是否需要随机打乱文件，由于不同光照的存在
        self.crop_size = calculate_valid_crop_size(128, 4)
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment = augment
        self.norm_flag = norm_flag
    def __getitem__(self, index):
        # input_image = load_img(self.image_filenames[index]) # 512 512 16 or 512 512 25
        input_image = np.load(self.image_filenames[index])
        input_image = input_image.astype(np.float32)
        if self.norm_flag:
            norm_name = 'maxnorm'
            max_raw = np.max(input_image)
            max_subband = np.max(np.max(input_image, axis=0), 0)
            norm_factor = max_raw / max_subband
            # ToDo 这里确认波段是16还是25
            # for bn in range(16):
            for bn in range(band_msc):
                input_image[:, :, bn] = input_image[:, :, bn] * norm_factor[bn]
        # print("input_image shape: ", input_image.shape)
        input_image = randcrop(input_image, self.crop_size)
        if self.augment:
            if np.random.uniform() < 0.5:
                input_image = np.fliplr(input_image)
            if np.random.uniform() < 0.5:
                input_image = np.flipud(input_image)
            # ToDo 增强方式是否足够
            input_image = np.rot90(input_image, k=np.random.randint(0, 4))
        target = input_image.copy()
        #ToDo 确认这里的mask
        ###原本的im_gt_y按照实际相机滤波阵列排列
        input_image = mask_input(target, mask)
        ###按照实际相机滤波阵列排列逆还原为从大到小的顺序
        input_image = reorder_imec(input_image) # sparase_image
        target = reorder_imec(target)  # multi-spectral

        if self.input_transform:
            raw = input_image.sum(axis=2)   #  how sum(axis=0 or 2)  2
            raw = self.input_transform(raw)  #this
            input_image = self.input_transform(input_image)   # sparase_image
            # print(input_image.size())
            # print(raw.size())
        if self.target_transform:
            # ToDo 这里确认波段是 25 还是 16
            target_PPI = target.sum(axis=2)/band_msc_f
            target_PPI = self.target_transform(target_PPI)
            # target = self.target_transform(target)/255.0

        return raw,target_PPI
        # return raw, input_image, target

    def __len__(self):
        return len(self.image_filenames)

if __name__ == '__main__':
    root = '/media/ssd1/zyg/'
    dataset = DatasetFromFolder(root)
    train_loader,_ = dataset.loaders(batch_size=32)
    for data in train_loader:
        mosaicked,ref = data
        print(mosaicked.shape)
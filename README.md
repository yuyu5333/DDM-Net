# DDM-Net
Please cite: A Deep Joint Network for Multispectral Demosaicking Based on Pseudo-Panchromatic Images

Shumin Liu, Yuge Zhang, Jie Chen, Keng-Pang Lim and Susanto Rahardja, "A Deep Joint Network for Multispectral Demosaicking Based on Pseudo-Panchromatic Images," in IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 4, pp. 622-635, Jun. 2022.

@article{liu2022deep,
  title={A Deep Joint Network for Multispectral Demosaicking Based on Pseudo-Panchromatic Images},
  author={Liu, Shumin and Zhang, Yuge and Chen, Jie and Lim, Keng Pang and Rahardja, Susanto},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={16},
  number={4},
  pages={622--635},
  year={2022},
  publisher={IEEE}
}

# 推理

```
python interfer_demosaic.py --model /home/dell/wyz/workGJS/DDM-Net/checkpoint/model_in_paper.pth --dataset /home/dell/wyz/workGJS/dataset/train_data/Yellow_Rose
```

```
CUDA_VISIBLE_DEVICES=0 python interfer_demosaic_less_memory.py --model /home/dell/wyz/workGJS/DDM-Net/checkpoint/model_in_paper.pth --dataset /home/dell/wyz/workGJS/dataset/TT31npy
```

```
CUDA_VISIBLE_DEVICES=2 python interfer_demosaic_less_mem_without_calcu.py --model /home/dell/wyz/workGJS/DDM-Net/checkpoint/model_in_paper.pth --dataset /home/dell/wyz/workGJS/dataset/TT31npy
```

# 训练PPI
```
CUDA_VISIBLE_DEVICES=0 python train_PPI.py --train_dir /home/dell/wyz/workGJS/dataset/CAVEnpy/CAVETrain --val_dir /home/dell/wyz/workGJS/dataset/CAVEnpy/CAVETest --batchSize 4

```

# 训练Model
```
CUDA_VISIBLE_DEVICES=0 python train_demosaic.py --train_dir /home/dell/wyz/workGJS/dataset/CAVEnpy/CAVETrain --val_dir /home/dell/wyz/workGJS/dataset/CAVEnpy/CAVETest --batchSize 4
```

# CAVE

## PPI

```
CUDA_VISIBLE_DEVICES=0 nohup python train_PPI.py --train_dir /home/dell/wyz/workGJS/dataset/CAVEnpy/CAVETrain --val_dir /home/dell/wyz/workGJS/dataset/CAVEnpy/CAVETest --batchSize 4 > /home/dell/wyz/workGJS/DDM-Net/log/CAVE/TrainPPIlog.txt 2>&1 &

```

## Model

```
CUDA_VISIBLE_DEVICES=2 nohup python train_demosaic.py --train_dir /home/dell/wyz/workGJS/dataset/CAVEnpy/CAVETrain --val_dir /home/dell/wyz/workGJS/dataset/CAVEnpy/CAVETest --batchSize 4 --PPI_pretrained /home/dell/wyz/workGJS/DDM-Net/checkpoint/PPI_Model/CAVE_PPI_model_epoch_3000.pth > /home/dell/wyz/workGJS/DDM-Net/log/CAVE/TrainModellog.txt 2>&1 &

```

## Fine Tuning
```
CUDA_VISIBLE_DEVICES=1 nohup python train_fine_tuning.py --train_dir /home/dell/wyz/workGJS/dataset/CAVEnpy/CAVETrain --val_dir /home/dell/wyz/workGJS/dataset/CAVEnpy/CAVETest --batchSize 4 --resume /home/dell/wyz/workGJS/DDM-Net/checkpoint/Model_Train/CAVE_main_model_epoch_2000.pth > /home/dell/wyz/workGJS/DDM-Net/log/CAVE/FineTrainModellog.txt 2>&1 &
```

# TT31

```
CUDA_VISIBLE_DEVICES=1 nohup python train_PPI.py --train_dir /home/dell/wyz/workGJS/dataset/TT31npy/TT31Train --val_dir /home/dell/wyz/workGJS/dataset/TT31npy/TT31Test --batchSize 4 > /home/dell/wyz/workGJS/DDM-Net/log/TT31/TrainPPIlog.txt 2>&1 &

```

## Model

```
CUDA_VISIBLE_DEVICES=0 nohup python train_demosaic.py --train_dir /home/dell/wyz/workGJS/dataset/TT31npy/TT31Train --val_dir /home/dell/wyz/workGJS/dataset/TT31npy/TT31Test --batchSize 4 --PPI_pretrained /home/dell/wyz/workGJS/DDM-Net/checkpoint/PPI_Model/TT31_PPI_model_epoch_3000.pth > /home/dell/wyz/workGJS/DDM-Net/log/TT31/TrainModellog.txt 2>&1 &

```

## Fine Tuning
```
CUDA_VISIBLE_DEVICES=0 nohup python train_fine_tuning.py --train_dir /home/dell/wyz/workGJS/dataset/TT31npy/TT31Train --val_dir /home/dell/wyz/workGJS/dataset/TT31npy/TT31Test --batchSize 4 --resume /home/dell/wyz/workGJS/DDM-Net/checkpoint/Model_Train/TT31_main_model_epoch_2000.pth --start-epoch 3001 > /home/dell/wyz/workGJS/DDM-Net/log/TT31/FineTrainModellog.txt 2>&1 &
```
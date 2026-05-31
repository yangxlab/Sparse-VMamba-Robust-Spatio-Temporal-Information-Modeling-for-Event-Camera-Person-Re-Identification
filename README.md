
1. Train
   ```python
     CUDA_VISIBLE_DEVICES=6 python train.py --represent  voxel --batchsize 32 --ReId_loss  softmax --AN_loss  SSIM --num_Bin 5  --epoch 60 --file_name  v1test --num_ids 22 --datasets full_data_v1 --gpu_ids 6
     CUDA_VISIBLE_DEVICES=6 python train.py --represent  voxel --batchsize 32 --ReId_loss  softmax --AN_loss  SSIM --num_Bin 5  --epoch 60 --file_name  v2test --num_ids 44 --datasets full_data_v2 --gpu_ids 6
```

2.Test
```python
   CUDA_VISIBLE_DEVICES=5 python test.py --model_path ./bestpath/tryv1nodrop/best_model_map.pth --datasets full_data_v1 --num_ids 22
   CUDA_VISIBLE_DEVICES=3 python test.py --model_path ./bestpath/tryv273/best_model_rank.pth --datasets full_data_v2 --num_ids 44
```


3. Bestpath
   Qurk Drive Link: https://pan.quark.cn/s/9d631a693c8e


4.Reference：
@article{dong2026sparse,
  title={Sparse VMamba: Robust Spatio-Temporal Information Modeling for Event Camera Person Re-Identification},
  author={Dong, Wenjiao and Yang, Xi and Wang, Nannan},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026},
  publisher={IEEE}
}

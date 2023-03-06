# KNN-OOD-Tao
The KNN-based out-of-distribution detection

Hyperparameters that may be used:

```python
parser.add_argument('--start_epoch', type=int, default=40, help='start epoch to use the outlier loss')
parser.add_argument('--num_layers', type=int, default=10, help='The number of layers to be fixed in CLIP')
parser.add_argument('--sample_number', type=int, default=1000, help='number of standard Gaussian noise samples')
parser.add_argument('--select', type=int, default=50, help='How many ID samples to pick to define as points near the boundary of the sample space')
parser.add_argument('--sample_from', type=int, default=1000, help='Number of IDs per class used to estimate OOD data.')
parser.add_argument('--T', type=int, default=10., help='temperature value')
parser.add_argument('--K', type=int, default=100, help='The value of top-K to calculate the KNN distance')
parser.add_argument('--loss_weight', type=float, default=0.1, help='The weight of outlier loss')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Learning rate decay ratio for MLP outlier')
parser.add_argument('--cov_mat', type=float, default=0.1, help='The weight before the covariance matrix to determine the sampling range')
parser.add_argument('--sampling_ratio', type=float, default=1., help='What proportion of points to choose to calculate the KNN value')
parser.add_argument('--ID_points_num', type=int, default=2, help='the number of synthetic outliers extracted for each selected ID')
parser.add_argument('--pick_nums', type=int, default=5, help='Number of ID samples used to generate outliers')

```

If you want to train a new model, you can execute:

```bash
python train_KNN.py --ngpu 8 --start_epoch 30 --sample_number 1000 --epochs 60 --sample_from 1500 --select 250 --loss_weight 0.1 --dataset ImageNet-100 --pick_nums 5 --cov_mat 0.1 --K 250 --save /nobackup-slow/taoleitian/model/vos/ImageNet-100/MCM/test/1 --batch_size 1600 --learning_rate 0.1 --decay_rate 0.1
```

If you want to test the trained model, you can execute:

```bash
python test_MCM.py --method_name ImageNet-100_dense_baseline_dense --score MSP --load /nobackup-slow/taoleitian/model/vos/ImageNet-100/MCM/test/1
```

If you want to test the the model trained by me, you can execute:
```bash
python test_MCM.py --method_name ImageNet-100_dense_baseline_dense --score MSP --load /nobackup-slow/taoleitian/model/vos/ImageNet-100/MCM/10/K/250
```

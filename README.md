# A PyTorch Implementation of MIPLMA

This is a PyTorch implementation of our paper "Multi-Instance Partial-Label Learning with Margin Adjustment", (**NeurIPS 2024**).

```bib
@inproceedings{tang2024miplma,
  author       = {Wei Tang and Yin-Fang Yang and Zhaofei Wang and Weijia Zhang and Min-Ling Zhang},
  title        = {Multi-Instance Partial-Label Learning with Margin Adjustment},
  booktitle    = {Advances in Neural Information  Processing Systems 37, Vancouver, Canada},
  year         = {2024},
}
```



## Requirements

```sh
numpy==1.21.5
scikit_learn==1.3.0
scipy==1.7.3
torch==1.12.0
```

To install the requirement packages, please run the following command:

```sh
pip install -r requirements.txt
```



## Reproductions

To reproduce the results of all datasets in the paper, please run the following commands:

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --ds MNIST_MIPL --ds_suffix 1 --nr_fea 784 --nr_class 5 --normalize false --lr 0.01 --epochs 100 --w_lambda 0.05 --min_tau 0.1
CUDA_VISIBLE_DEVICES=0 python main.py --ds MNIST_MIPL --ds_suffix 2 --nr_fea 784 --nr_class 5 --normalize false --lr 0.01 --epochs 100 --w_lambda 0.1 --min_tau 0.1
CUDA_VISIBLE_DEVICES=0 python main.py --ds MNIST_MIPL --ds_suffix 3 --nr_fea 784 --nr_class 5 --normalize false --lr 0.01 --epochs 100 --w_lambda 0.01 --min_tau 0.1
```



If you are interested in multi-instance partial-label learning, [MIPLGP](https://tangw-seu.github.io/publications/SCIS'23.pdf), [DEMIPL](https://tangw-seu.github.io/publications/NeurIPS'23.pdf), [ELIMIPL](https://tangw-seu.github.io/publications/IJCAI'24.pdf), and [ProMIPL](https://tangw-seu.github.io/publications/ICDM'24.pdf) may be helpful to you.

This package is only free for academic usage. Have fun!
# MORI-RAN: Multi-view Robust Representation Learning via Hybrid Contrastive Fusion
The official repos. for "MORI-RAN: Multi-view Robust Representation Learning via Hybrid Contrastive Fusion"

- Submitted at: ICDM 2022 Workshop on Multi-view Representation Learning

- Status: Under Review.

- [ArXiv](https://arxiv.org/abs/2208.12545)


## Datasets

The Caltech101-20, LandUse-21, and Scene-15 datasets are placed in `src/data/processed` folder. 

The NoisyMNIST dataset could be downloaded from [Google Cloud](https://drive.google.com/file/d/1b__tkQMHRrYtcCNi_LxnVVTwB-TWdj93/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1qcIkZtVCb26GCa0wglJNBw) **password: 9q5e**.

Thanks for [COMPLETER](https://github.com/XLearning-SCU/2021-CVPR-Completer).


## Quickly start

If want to train the model with `fp16`, you need to install the `Apex`.

- Step 1: install dependencies.

```
git clone https://github.com/Guanzhou-Ke/mori-ran.git
cd mori-ran
conda create -n moriran --file requirements.txt
conda activate moriran
```

- Step 2: run MORIRAN on `NoisyNMIST`

```
cd src
python main.py -d 0
```

If want to run `CONAN`:
```
python main.py -e conan -d 0
```

Finally, all experimental results will save at `./src/experiments/results` 

## Advance experiment

You can modify the configuration files at `./src/experiments/configs`. We employ the [YACS](https://github.com/rbgirshick/yacs) style to write the experimental configs.

## Common issues

- (Apex) The issue of "tuple index out of range" from cached_cast.

Modify the `apex/amp/utils.py#cached_cast` as following:

```
# change this line (line 113)
- if cached_x.grad_fn.next_functions[1][0].variable is not x:
# into this
+ if cached_x.grad_fn.next_functions[0][0].variable is not x:
```

[Issue Link](https://github.com/NVIDIA/apex/issues/694)

## Citation

```
@misc{ke2022moriran,
    title={MORI-RAN: Multi-view Robust Representation Learning via Hybrid Contrastive Fusion},
    author={Guanzhou Ke and Yongqi Zhu and Yang Yu},
    year={2022},
    eprint={2208.12545},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
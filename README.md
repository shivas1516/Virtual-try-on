# Virtual Try-On Project

## Overview
This project aims to learn, explore, and implement various deep-learning models for virtual try-on applications. It utilizes techniques such as GFLA, PyTorch-CycleGAN-and-pix2pix, PATN, and MUNIT for generating virtual try-on results.

## Demo
A runnable demo can be accessed in our Colab environment. [Open In Colab!](https://colab.research.google.com/)

## Getting Started with Bash Scripts

### DeepFashion Dataset Setup
1. **Download DeepFashion Dataset:**
    ```bash
    pip install --upgrade gdown
    python tools/download_deepfashion_from_google_drive.py --dataroot $DATA_ROOT
    ```
    This script downloads necessary data from Google Drive and organizes it in the specified `$DATA_ROOT` directory.

### Environment Setup

1. **Environment for Inference or Test:**
    ```bash
    pip install torch torchvision tensorboardX scikit-image==0.16.2
    ```
    This installs the required packages for inference.

2. **Environment for Training:**
    Training requires CUDA functions provided by GFLA, which compiles with `torch=1.0.0`. Follow installation instructions in GFLA and then run:
    ```bash
    pip install -r requirements.txt
    ```

### Download Pretrained Weights
Pretrained weights can be downloaded [here](#). Unzip them under the `checkpoints/` directory.

### Training
1. **Warmup the Global Flow Field Estimator:**
    Run:
    ```bash
    sh scripts/run_pose.sh
    ```
    Note: If you prefer not to warm up the estimator, you can extract its weights from GFLA.

2. **Training:**
    After warming up, train the pipeline with:
    ```bash
    sh scripts/run_train.sh
    ```
    Monitor training progress with:
    ```bash
    tensorboard --logdir checkpoints/$EXP_NAME/train
    ```
    Note: Resetting discriminators may help when training gets stuck at local minima.

### Evaluations
1. **Download Generated Images:**
    Access generated images used for evaluation [here](https://drive.google.com/drive/folders/1-7DxUvcrC3cvQV67Z2QhRdi-9PMDC8w9?usp=sharing).

2. **SSIM, FID, and LPIPS:**
    To run evaluation (SSIM, FID, and LPIPS) on pose transfer task:
    ```bash
    sh scripts/run_eval.sh
    ```
    Always specify `--frozen_flownet` for inference.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## References
- This repository is built on [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention), [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [PATN](https://github.com/tengteng95/Pose-Transfer), and [MUNIT](https://github.com/NVlabs/MUNIT). Please be aware of their licenses when using the code.

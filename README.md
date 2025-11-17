# BME 2025 Deep Learning Project
## Enhancing Image Super-Resolution with Generative Adversarial Networks (GANs)

### Members
Team VEO
- Vince Szigetvari OWC5ZP
- Eduardo Meza Medina IKUL3K
- Onur Aktan H55CGX

---
## Aim of the project

Image Super-Resolution (ISR) is the process of enhancing the resolution of low-quality images to improve detail and clarity. This project will explore the use of GANs, particularly architectures like SRGAN (Super-Resolution GAN) or ESRGAN (Enhanced SRGAN), to upscale low-resolution images while maintaining or improving perceptual quality.

We will study the ISR domain and implement a GAN-based model (e.g., SRGAN or ESRGAN). Furthermore, we will train it on DIV2K dataset to generate higher-resolution images from lower-resolution inputs. Our focus will be on comparing traditional interpolation methods (e.g., bilinear or bicubic) against GAN-based methods in terms of image quality (measured by PSNR and SSIM) and perceptual realism (using MOS scores). First, we will implement SRGAN and train it on DIV2K dataset for super-resolving images at a standard scale (e.g., 2x upscaling). Secondly, we will experiment with ESRGAN and additional techniques to improve the super-resolution quality. 

---
## Presentation of dataset

We are using the [DIV2K dataset](https://www.kaggle.com/datasets/francescopignatelli/div2k-dataset-antialias?resource=download-directory) for image super-resolution tasks. The dataset consists of high-quality images and their corresponding low-resolution versions, which are used for training and evaluating super-resolution models. Since the dataset at the original source was not available to download, we downloaded it from Kaggle.

The dataset has the following structure:
```
Dataset/
    ├── train/
    ├── train_labels/
    ├── validation/
    ├── validation_labels/
    ├── test/
    └── test_labels/
```

#### Downloading the dataset

The dataset can be downloaded as zip file from [Kaggle](https://www.kaggle.com/datasets/francescopignatelli/div2k-dataset-antialias?resource=download-directory). After downloading, unzip the file and place the extracted `Dataset` folder in the root directory of this project.

#### Dataset details

There are 19570 train image-pairs, 4193 validation image-pairs and 4195 test image-pairs in total. The size of the LR images is 64x64, and 256*256 for HR images. Originally LR images were 256x256 (but more blurry), but we downscaled them to 64x64 using `cv2.INTER_CUBIC` to create a more challenging super-resolution task. The script used for downscaling can be found in `downsize_inputs.py`. The dataset with dowsized images can be found on [Google Drive](https://drive.google.com/file/d/1lqBvhWfWsF_F2siT6PUEvgi3Eiboj5lC/view?usp=drive_link).

Exploration of the dataset can be found in the `data_exploration.ipynb` notebook.

![input_10 / output_10](readme_assets/example10.png)

---
## How to run project

There are 2 possibilities to run the projec.

#### Run project using Google Colab

This way is recommended if you do not have a powerful GPU available locally. Both training and evaluation can be done in the following [colab notebook](https://colab.research.google.com/drive/1qBS-JGl1o3cseP8tEqf0Va77MEKkgHdg?usp=sharing).

#### Run project locally

1. Clone the [repository](https://github.com/Antioksidan/DeepLearning) to your local machine. 

```bash
git clone https://github.com/Antioksidan/DeepLearning
cd DeepLearning
```

2. Make sure you have Python 3.8+ installed. It is recommended to use a virtual environment. You can create and activate a virtual environment using the following commands:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

4. Dependencies of the projects are in the src folder:

```
src/
    ├── __init__.py
    ├── dataloader.py # data loading and preprocessing
    ├── discriminator.py # discriminator model
    ├── generator.py # generator model
    ├── training_functions.py # training loops
    └── vgg_wrapper.py # VGG feature extractor for perceptual loss
```

5. For training the models, you can run the `train_pipeline.ipynb` notebook. Make sure to adjust the paths to the dataset if necessary.

6. For evaluating the trained models, you can run the `eval_pipeline.ipynb` notebook. Again, ensure that the paths to the dataset and model checkpoints are correctly set.
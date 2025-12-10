# BME 2025 Deep Learning Project
## Enhancing Image Super-Resolution with Generative Adversarial Networks (GANs)

### Members
Team VEO
- Vince Szigetvari OWC5ZP
- Eduardo Meza Medina IKUL3K
- Onur Aktan H55CGX

---
# 1. Introduction

TODO:

kkklllmmmmjkjhlk

o	Overview of the chosen topic

o	References to relevant scientific papers


---
# 2. Methods

## 2.1 SRGAN

TODO: describing GAN and superresolution concepts, and SRGAN architecture from the paper

### Discriminator Network

TODO: describing Discriminator architecture from srgan paper

![discriminator](readme_assets/discriminator.png)

### Generator Network

TODO: describing Generator architecture from srgan paper (8 RB instead of 16 in paper to make training faster)

![generator](readme_assets/generator.png)

### Loss Functions

TODO: describing loss functions from srgan paper

## 2.2 ESRGAN

### Modified Generator Network

TODO: describing ESRGAN generator architecture from the paper (what changed compared to SRGAN)

![esrgan_generator](readme_assets/generator2.png)

![rrdb](readme_assets/rrdb.png)

### Modified Loss Functions

TODO: describing loss functions from esrgan paper (what changed compared to SRGAN, (VGG loss before relu, Discriminator loss changes))

---
# 3. Training

TODO: describing training, times, hardware (L4 GPU on colab), tracking loss on wandb, changing hyperparameters (mainly the weight for the adversarial loss)

---
# 4. Evaluation

o	Results on the test data (metrics in table (PSNR and VGG loss))

o	Visualizations (plots, metrics, comparisons, etc.)

o	Sometimes subjective evaluation is also useful (e.g., user studies), based on our experience, ESRGAN had worse metrics but looked better visually

![example_0](readme_assets/result_pics/example_0.png)

![example_1](readme_assets/result_pics/example_1.png)

![example_2](readme_assets/result_pics/example_2.png)

![example_4](readme_assets/result_pics/example_4.png)

![example_5](readme_assets/result_pics/example_5.png)

![example_6](readme_assets/result_pics/example_6.png)

![example_7](readme_assets/result_pics/example_7.png)

![example_8](readme_assets/result_pics/example_8.png)

![example_9](readme_assets/result_pics/example_9.png)

![example_10](readme_assets/result_pics/example_10.png)

![example_11](readme_assets/result_pics/example_11.png)

![example_13](readme_assets/result_pics/example_13.png)

![example_14](readme_assets/result_pics/example_14.png)

![example_15](readme_assets/result_pics/example_15.png)

![example_18](readme_assets/result_pics/example_18.png)

![example_20](readme_assets/result_pics/example_20.png)

![example_21](readme_assets/result_pics/example_21.png)

![example_22](readme_assets/result_pics/example_22.png)

![example_25](readme_assets/result_pics/example_25.png)

![example_26](readme_assets/result_pics/example_26.png)

![example_27](readme_assets/result_pics/example_27.png)

![example_29](readme_assets/result_pics/example_29.png)

![example_30](readme_assets/result_pics/example_30.png)

![example_31](readme_assets/result_pics/example_31.png)

![example_33](readme_assets/result_pics/example_33.png)

![example_34](readme_assets/result_pics/example_34.png)

---
# 5. Conclusions

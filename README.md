# CF-CLIP (Towards Counterfactual Image Manipulation via CLIP)
This repository is an official PyTorch implementation of the ACM MM 2022 paper "Towards Counterfactual Image Manipulation via CLIP".
## Setup
The code relies on the official implementation of [CLIP](https://github.com/openai/CLIP), 
and the [Rosinality](https://github.com/rosinality/stylegan2-pytorch/) pytorch implementation of StyleGAN2.

### Requirements
For all the methods described in the paper, is it required to have:
- Anaconda
- [CLIP](https://github.com/openai/CLIP)

Specific requirements for each method are described in its section. 
To install CLIP please run the following commands:
  ```shell script
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git
```
## Pretrained Models
Please download the following pertrained models and place them in `./pretrained` folder.
### StyleGAN
- [FFHQ](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing)
- [AFHQ Dog](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/afhqdog.pkl)
- [AFHQ Cat](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/afhqcat.pkl)

For AFHQ Dog and Cat, we can convert the tensorflow version pretrained model to pytorch version using `convert_weight.py`.

### Face Recognition & VGG 
- [Face Recognition](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing)
- [VGG](https://drive.google.com/file/d/1fp7DAiXdf0Ay-jANb8f0RHYLTRyjNv4m/view?usp=sharing)

## Latent Codes
### Inverted CelebA-HQ via e4e:
- [train set](https://drive.google.com/file/d/1gof8kYc_gDLUT4wQlmUdAtPnQIlCO26q/view?usp=sharing)
- [test set](https://drive.google.com/file/d/1j7RIfmrCoisxx3t-r-KC02Qc8barBecr/view?usp=sharing)
### Random Sample
For cat and dog, we randomly sample w code (1*512) using `GetCode.py`, which uses the tensorflow version pretrained model. ('.pkl'). In this case, we need to set `w_space` option of training script to `True`.

## Usage
### Pretrained Models

We provided pretrained models for different face, AFHQ Dog and Cat cases in our paper [here](https://drive.google.com/drive/folders/1GxwCtTCgVG9nUkkVTOTyES7lWw5zhHQ7?usp=sharing). You may put them under folder `pretrained` after downloading.

### Training
- The main training script is placed in `mapper/scripts/train.py`.
- Training arguments can be found at `mapper/options/train_options.py`.
- Intermediate training results are saved to opts.exp_dir. This includes checkpoints, train outputs, and test outputs.
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in opts.exp_dir/logs.
Note that
- To resume a training, please provide `--checkpoint_path`.
- `--description` is where you provide the driving text.

Example for training a mapper for the green lipstick:
```bash
cd mapper
python scripts/train.py --exp_dir ../results/green_lipstick --description "green lipstick"
```
You may refer `train.sh` for the example of training AFHQ Dog/Cat cases.


### Inference
- The main inferece script is placed in `mapper/scripts/inference.py`.
- Inference arguments can be found at `mapper/options/test_options.py`.
- Adding the flag `--couple_outputs` will save image containing the input and output images side-by-side.

You may refer `test.sh` for reference.

## Citation

If you find CF-CLIP useful or inspiring, please consider citing:

```bibtex
@inproceedings{yu2022-CFCLIP,
  title       = {Towards Counterfactual Image Manipulation via CLIP},
  author      = {Yu, Yingchen and Zhan, Fangneng and Wu, Rongliang and Zhang, Jiahui and Lu, Shijian and Cui, Miaomiao and Xie, Xuansong and Hua, Xian-Sheng and Miao, Chunyan},
  booktitle   = {Proceedings of the 30th ACM International Conference on Multimedia},
  year        = {2022}
}
```

## Acknowledgments
This code borrows heavily from [StyleCLIP](https://github.com/orpatashnik/StyleCLIP), [StyleGAN-NADA](https://github.com/rinongal/StyleGAN-nada) and [InfoNCE](https://github.com/RElbers/info-nce-pytorch), we apprecite the authors for sharing their codes.
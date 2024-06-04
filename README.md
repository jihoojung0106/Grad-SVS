# Grad-SVS
This project performs a Korean singing voice synthesis task using a diffusion model, specifically based on Grad-TTS.
More details and demo at [here](https://jihoojung0106.github.io/posts/Grad_SVS/)

## Installation

```bash
pip install -r requirements.txt
```
## Dataset

We used the [Children Song Dataset](https://github.com/emotiontts/emotiontts_open_db/tree/master/Dataset/CSD), an open-source singing voice dataset comprised of 100 annotated Korean and English children songs sung by a single professional singer. We used only the Korean subset of the dataset to train the model.

You can train the model on any custom dataset of your choice, as long as it includes lyrics text, midi transcriptions, and monophonic a capella audio file triplets. These files should be titled identically, and should also be placed in specific directory locations as shown below.

```
├── data
│   └── raw
│       ├── mid
│       ├── txt
│       └── wav
```

The directory names correspond to file extensions. We have included a sample as reference.

## Preprocessing

Once you have prepared the dataset, run 

```
cd data_preprocess
python serialize.py
```

from the root directory. This will create `data_preprocess/bin` that contains binary files used for training. This repository already contains example binary files created from the sample in `data_preprocess/raw`. 

## Inference

```bash
my_inference.py
```
To generate audio files with the trained model checkpoint, [download](https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd) the HiFi-GAN checkpoint along with its configuration file and place them in `hifi-gan-mlp`. 

## Training
```bash
train.py
```
## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Grad-TTS : [link](https://github.com/neosapience/mlp-singer).
* Data preprocessing: [link](https://github.com/neosapience/mlp-singer).



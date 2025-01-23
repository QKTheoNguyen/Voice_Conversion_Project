# Voice Conversion Project using StarGANv2-VC

Welcome to this Voice Conversion Project forked from [StarGANv2-VC](https://github.com/yl4579/StarGANv2-VC) repository, developed by Yinghao Aaron Li, Ali Zare, and Nima Mesgarani.

# Overview

This project aims to train StarGANv2-VC model on a dataset of french speakers to convert the voice of a source speaker to a target speaker.

# Dataset

The dataset used for this project is a part of the [Common Voice](https://commonvoice.mozilla.org/fr/datasets) dataset, which is a multilingual dataset of voices that is publicly available for research purposes. 

The dataset contains the Delta segments 17.0, 18.0, 19.0 and 20.0 of the Common Voice french corpus.

# Preprocessing

The dataset is preprocessed using the following steps:
- Extract the audio files from the dataset
- Extract metadata from the dataset
- Selection of the speakers to be used in the training (based on the number of utterances) and grouping them into directories
- Cutting the audio files on speaking segments using a Voice Activity Detection algorithm (Silero VAD) and resampling them to 16kHz


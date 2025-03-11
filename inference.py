import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import argparse

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder

path_jdc = "/content/Voice_Conversion_Project/Utils/JDC/bst.t7"
path_vocoder = "/content/Voice_Conversion_Project/Vocoder/checkpoint-400000steps.pkl"
model_path = "/content/drive/MyDrive/Voice_Conversion_Dataset/Models/common_voice_fr/epoch_00150.pth"
path_config = "/content/drive/MyDrive/Voice_Conversion_Dataset/Models/common_voice_fr/config.yml"

# mel spectrogram transformation

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

# audio processing

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

# build models

def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema

def compute_style(speaker_dicts, starganv2):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == "":
            label = torch.LongTensor([speaker]).to('cuda')
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(torch.randn(1, latent_dim).to('cuda'), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave).to('cuda')

            with torch.no_grad():
                label = torch.LongTensor([speaker])
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)
    
    return reference_embeddings

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Models/common_voice/epoch_00070.pth")
    parser.add_argument("--config", type=str, default="Models/common_voice/config.yaml")
    parser.add_argument("--audio_file", type=str, default="demo")
    parser.add_argument("--speaker1", type=int)
    parser.add_argument("--speaker2", type=int)
    parser.add_argument("--output_file", type=str, default="output.wav")
    args = parser.parse_args()

    ### load models

    # load F0 model

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path_jdc, weights_only=True)['net']
    F0_model.load_state_dict(params)
    _ = F0_model.eval()
    F0_model = F0_model.to('cuda')

    # load vocoder
    from parallel_wavegan.utils import load_model
    vocoder = load_model(path_vocoder).to('cuda').eval()
    vocoder.remove_weight_norm()
    _ = vocoder.eval()

    # load starganv2

    with open(path_config) as f:
        starganv2_config = yaml.safe_load(f)
    starganv2 = build_model(model_params=starganv2_config["model_params"])
    params = torch.load(model_path, map_location='cpu')
    params = params['model_ema']
    _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    _ = [starganv2[key].eval() for key in starganv2]
    starganv2.style_encoder = starganv2.style_encoder.to('cuda')
    starganv2.mapping_network = starganv2.mapping_network.to('cuda')
    starganv2.generator = starganv2.generator.to('cuda')

    speakers = [f"{i:03d}" for i in range(2, 21)]
    speaker_dicts = {}

    ### load input audio

    if args.audio_file != "demo":

        audio, source_sr = librosa.load(args.audio_file, sr=24000)
        audio = audio / np.max(np.abs(audio))
        audio.dtype = np.float32

        selected_speaker = args.speaker1
        target_speaker = args.speaker2
        speaker_dicts[selected_speaker] = {(args.audio_file, speakers.index(selected_speaker))}

        reference_embeddings = compute_style(speaker_dicts, starganv2)

        source = preprocess(audio).to('cuda:0')
        keys = []
        converted_samples = {}
        reconstructed_samples = {}
        converted_mels = {}

        for key, (ref, _) in reference_embeddings.items():
            with torch.no_grad():
                f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
                out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)
                
                c = out.transpose(-1, -2).squeeze().to('cuda')
                y_out = vocoder.inference(c)
                y_out = y_out.view(-1).cpu()

                if key not in speaker_dicts or speaker_dicts[key][0] == "":
                    recon = None
                else:
                    wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
                    mel = preprocess(wave)
                    c = mel.transpose(-1, -2).squeeze().to('cuda')
                    recon = vocoder.inference(c)
                    recon = recon.view(-1).cpu().numpy()

            converted_samples[key] = y_out.numpy()
            reconstructed_samples[key] = recon

            converted_mels[key] = out
            
            keys.append(key)

        for key, wave in converted_samples.items():
            print('Converted: %s' % key)
            if reconstructed_samples[key] is not None:
                librosa.output.write_wav(args.output_file, reconstructed_samples[key], 24000)

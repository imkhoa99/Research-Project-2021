# =====================
# Quick script to preprocess audio files and set them up
# For the jukemir representations processing.
#
# This mostly pads the audio files to 24 secs long.
# =====================

import torch
import torchaudio
import torch.nn as nn
import torch.functional as F

from tqdm import tqdm
import os


def pad_audio(audio: torch.Tensor, sampling_rate: int,
              target_len_seconds: float = 24.0, padding_mode='zeros') -> torch.Tensor:
    """ Audio should be a batch of audio tensors
    With shapes [batch, channel, timesteps]"""
    target_len = target_len_seconds * sampling_rate
    assert len(audio.shape) == 2, 'Wrong shape for audio tensor. Should be [channel, timesteps]'
    pad_size = (target_len - audio.shape[-1]) / 2
    pad_size = int(pad_size)
    if padding_mode == 'zeros':
        padder = nn.ConstantPad1d(padding=pad_size, value=0.0)
    elif padding_mode == 'reflection':  # Reverse in time
        padder = nn.ReflectionPad1d(padding=pad_size)
    elif padding_mode == 'replication':
        padder = nn.ReplicationPad1d(padding=pad_size)

    return padder(audio)


def load_audio(fname: str) -> torch.Tensor:
    audio, fs = torchaudio.load(fname)
    return audio, fs


# From vrgpu :
# data_path: /m/cs/cs/sequentialml/datasets/audio-visual/audio
def get_config():
    config = {
        'data_path': '/m/cs/cs/sequentialml/datasets/audio-visual/audio/',
        'output_path': '/m/cs/cs/sequentialml/datasets/audio-visual/audio_padded/',
        'job_chunk_size': 12291,
        'job_chunk_id': 0,
        'mode': 'zeros',
    }

    return config


def process():
    config = get_config()

    directory = os.fsencode(config['data_path'])
    start_id = config['job_chunk_size'] * config['job_chunk_id']
    end_id = config['job_chunk_size'] * (config['job_chunk_id'] + 1)

    print('Processing chunks with files ids:')
    print(f'start_id: {start_id}')
    print(f'start_id: {end_id}')
    print(config)

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
    with tqdm(total=config['job_chunk_size'] ) as pbar:
        for ctr, file in enumerate(os.listdir(directory)):
            if ctr < start_id:
                continue
            if ctr > end_id:
                break
            fname = os.fsdecode(file)
            if fname.endswith(".wav") or fname.endswith(".mp3"):
                audio, fs = load_audio(config['data_path'] + fname)
                padded = pad_audio(audio, fs, target_len_seconds=24, padding_mode=config['mode'])
                torchaudio.save(config['output_path'] + fname, padded, fs)
            pbar.update(1)

    print('finished')

if __name__ == '__main__':
    process()




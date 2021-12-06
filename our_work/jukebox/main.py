#%%
import librosa as lr
import numpy as np
import torch

JUKEBOX_SAMPLE_RATE = 44100

def load_audio_from_file(fpath):
    audio, _ = lr.load(fpath, sr=JUKEBOX_SAMPLE_RATE)
    if audio.ndim == 1:
        audio = audio[np.newaxis]
    audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    return audio.flatten()
    
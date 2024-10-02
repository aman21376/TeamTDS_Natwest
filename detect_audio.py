# detect_audio.py

import numpy as np
import os
import logging
from models import load_wavefake_model, load_resnet_audio_model
from preprocessing import preprocess_audio_waveform, preprocess_audio_spectrogram
from utils import generate_report
from config import AUDIO_OUTPUT_DIR

def detect_audio_deepfake(audio_path):
    logging.info(f'Starting deepfake detection on audio: {audio_path}')
    if not os.path.exists(audio_path):
        logging.error(f'Audio file {audio_path} does not exist.')
        return {'confidence': 0.0, 'report': 'Audio file not found.'}
    try:
        wavefake_model = load_wavefake_model()
        resnet_audio_model = load_resnet_audio_model()
    except Exception as e:
        logging.error(f'Error loading models: {e}')
        return {'confidence': 0.0, 'report': 'Error loading models.'}
    waveform = preprocess_audio_waveform(audio_path)
    if waveform is None:
        logging.error(f'Failed to preprocess audio waveform {audio_path}.')
        return {'confidence': 0.0, 'report': 'Failed to preprocess audio waveform.'}
    spectrogram = preprocess_audio_spectrogram(audio_path)
    if spectrogram is None:
        logging.error(f'Failed to preprocess audio spectrogram {audio_path}.')
        return {'confidence': 0.0, 'report': 'Failed to preprocess audio spectrogram.'}
    try:
        wavefake_pred = wavefake_model.predict(waveform)
        wavefake_confidence = wavefake_pred[0][0]
        logging.debug(f'WaveFake confidence: {wavefake_confidence}')
    except Exception as e:
        logging.error(f'Error during WaveFake prediction: {e}')
        wavefake_confidence = 0.0
    try:
        spectrogram_pred = resnet_audio_model.predict(spectrogram)
        spectrogram_confidence = spectrogram_pred[0][0]
        logging.debug(f'Spectrogram confidence: {spectrogram_confidence}')
    except Exception as e:
        logging.error(f'Error during spectrogram prediction: {e}')
        spectrogram_confidence = 0.0
    combined_confidence = (wavefake_confidence + spectrogram_confidence) / 2.0
    logging.info(f'Combined confidence score: {combined_confidence}')
    report = generate_report(combined_confidence, 'audio')
    return {
        'confidence': combined_confidence,
        'report': report
    }

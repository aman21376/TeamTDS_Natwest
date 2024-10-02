# detect_video.py

import cv2
import numpy as np
import os
import logging
from models import load_xception_model, load_conv_lstm_model
from preprocessing import extract_video_frames, preprocess_frames
from utils import generate_report, save_video_with_highlights
from config import VIDEO_OUTPUT_DIR

def detect_video_deepfake(video_path):
    logging.info(f'Starting deepfake detection on video: {video_path}')
    if not os.path.exists(video_path):
        logging.error(f'Video file {video_path} does not exist.')
        return {'confidence': 0.0, 'report': 'Video file not found.'}
    try:
        xception_model = load_xception_model()
        conv_lstm_model = load_conv_lstm_model()
    except Exception as e:
        logging.error(f'Error loading models: {e}')
        return {'confidence': 0.0, 'report': 'Error loading models.'}
    frames = extract_video_frames(video_path)
    if frames is None or len(frames) == 0:
        logging.error(f'Failed to extract frames from video {video_path}.')
        return {'confidence': 0.0, 'report': 'Failed to extract video frames.'}
    preprocessed_frames = preprocess_frames(frames)
    try:
        spatial_preds = xception_model.predict(preprocessed_frames)
        spatial_confidence = np.mean(spatial_preds)
        logging.debug(f'Spatial confidence: {spatial_confidence}')
    except Exception as e:
        logging.error(f'Error during spatial prediction: {e}')
        spatial_confidence = 0.0
    try:
        preprocessed_frames_seq = np.expand_dims(preprocessed_frames, axis=0)
        temporal_pred = conv_lstm_model.predict(preprocessed_frames_seq)
        temporal_confidence = temporal_pred[0][0]
        logging.debug(f'Temporal confidence: {temporal_confidence}')
    except Exception as e:
        logging.error(f'Error during temporal prediction: {e}')
        temporal_confidence = 0.0
    combined_confidence = (spatial_confidence + temporal_confidence) / 2.0
    logging.info(f'Combined confidence score: {combined_confidence}')
    report = generate_report(combined_confidence, 'video')
    output_video_path = os.path.join(VIDEO_OUTPUT_DIR, os.path.basename(video_path))
    save_video_with_highlights(frames, spatial_preds, output_video_path)
    logging.info(f'Processed video saved to {output_video_path}')
    return {
        'confidence': combined_confidence,
        'report': report
    }

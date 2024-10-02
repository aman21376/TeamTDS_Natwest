# config.py

import os

# Paths to models
XCEPTION_MODEL_PATH = 'models/xception_model.h5'
FACE_XRAY_MODEL_PATH = 'models/face_xray_model.h5'
CONV_LSTM_MODEL_PATH = 'models/conv_lstm_model.h5'
WAVEFAKE_MODEL_PATH = 'models/wavefake_model.h5'
RESNET_AUDIO_MODEL_PATH = 'models/resnet_audio_model.h5'

# Preprocessing parameters
IMAGE_SIZE = (299, 299)
AUDIO_SAMPLE_RATE = 16000
NUM_VIDEO_FRAMES = 20

# Thresholds
THRESHOLD = 0.7

# Logging level
LOGGING_LEVEL = 'INFO'

# Output directories
IMAGE_OUTPUT_DIR = 'output_images'
VIDEO_OUTPUT_DIR = 'output_videos'
AUDIO_OUTPUT_DIR = 'output_audio'

# Create output directories if they don't exist
if not os.path.exists(IMAGE_OUTPUT_DIR):
    os.makedirs(IMAGE_OUTPUT_DIR)
if not os.path.exists(VIDEO_OUTPUT_DIR):
    os.makedirs(VIDEO_OUTPUT_DIR)
if not os.path.exists(AUDIO_OUTPUT_DIR):
    os.makedirs(AUDIO_OUTPUT_DIR)

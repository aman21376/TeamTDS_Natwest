# preprocessing.py

import cv2
import numpy as np
import librosa
from tensorflow.keras.applications.xception import preprocess_input
from config import IMAGE_SIZE, AUDIO_SAMPLE_RATE, NUM_VIDEO_FRAMES

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, IMAGE_SIZE)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def extract_video_frames(video_path, num_frames=NUM_VIDEO_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, IMAGE_SIZE)
            frame = preprocess_input(frame)
            frames.append(frame)
        else:
            break
    cap.release()
    return np.array(frames)

def preprocess_frames(frames):
    return frames

def preprocess_audio_waveform(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
        y = y[:AUDIO_SAMPLE_RATE]
        y = np.expand_dims(y, axis=0)
        y = np.expand_dims(y, axis=-1)
        return y
    except Exception as e:
        return None

def preprocess_audio_spectrogram(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        log_spectrogram -= log_spectrogram.min()
        log_spectrogram /= log_spectrogram.max()
        log_spectrogram = np.expand_dims(log_spectrogram, axis=0)
        log_spectrogram = np.expand_dims(log_spectrogram, axis=-1)
        return log_spectrogram
    except Exception as e:
        return None


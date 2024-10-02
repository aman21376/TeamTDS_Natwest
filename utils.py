# utils.py

import cv2
import numpy as np
import os
import logging
from config import THRESHOLD

def generate_report(confidence, media_type):
    if confidence > THRESHOLD:
        status = 'High likelihood of being a deepfake.'
    elif confidence > THRESHOLD / 2:
        status = 'Potential deepfake detected.'
    else:
        status = 'Media appears authentic.'
    report = f"The analyzed {media_type} has a confidence score of {confidence:.2f}. {status}"
    return report

def highlight_suspicious_regions(image, mask):
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_normalized = (mask_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
    highlighted_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return highlighted_image

def save_video_with_highlights(frames, predictions, output_path):
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    for i, frame in enumerate(frames):
        confidence = predictions[i][0]
        if confidence > THRESHOLD:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        out.write(frame)
    out.release()


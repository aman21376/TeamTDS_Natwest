# detect_image.py

import cv2
import numpy as np
import os
import logging
from models import load_xception_model, load_face_xray_model
from preprocessing import preprocess_image
from utils import generate_report, highlight_suspicious_regions
from config import IMAGE_OUTPUT_DIR, THRESHOLD, IMAGE_SIZE

def detect_image_deepfake(image_path):
    logging.info(f'Starting deepfake detection on image: {image_path}')
    if not os.path.exists(image_path):
        logging.error(f'Image file {image_path} does not exist.')
        return {'confidence': 0.0, 'report': 'Image file not found.'}
    try:
        xception_model = load_xception_model()
        face_xray_model = load_face_xray_model()
    except Exception as e:
        logging.error(f'Error loading models: {e}')
        return {'confidence': 0.0, 'report': 'Error loading models.'}
    img = preprocess_image(image_path)
    if img is None:
        logging.error(f'Failed to preprocess image {image_path}.')
        return {'confidence': 0.0, 'report': 'Failed to preprocess image.'}
    try:
        xception_pred = xception_model.predict(img)
        xception_confidence = xception_pred[0][0]
        logging.debug(f'Xception model confidence: {xception_confidence}')
    except Exception as e:
        logging.error(f'Error during Xception model prediction: {e}')
        xception_confidence = 0.0
    try:
        manipulated_mask = face_xray_model.predict(img)[0, :, :, 0]
        manipulated_score = np.mean(manipulated_mask)
        logging.debug(f'Face X-ray model manipulated score: {manipulated_score}')
    except Exception as e:
        logging.error(f'Error during Face X-ray model prediction: {e}')
        manipulated_score = 0.0
        manipulated_mask = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1]))
    combined_confidence = (xception_confidence + manipulated_score) / 2.0
    logging.info(f'Combined confidence score: {combined_confidence}')
    report = generate_report(combined_confidence, 'image')
    original_img = cv2.imread(image_path)
    if original_img is not None:
        highlighted_img = highlight_suspicious_regions(original_img, manipulated_mask)
        if not os.path.exists(IMAGE_OUTPUT_DIR):
            os.makedirs(IMAGE_OUTPUT_DIR)
        output_image_path = os.path.join(IMAGE_OUTPUT_DIR, os.path.basename(image_path))
        cv2.imwrite(output_image_path, highlighted_img)
        logging.info(f'Highlighted image saved to {output_image_path}')
    else:
        logging.error(f'Failed to read original image {image_path}')
    return {
        'confidence': combined_confidence,
        'report': report
    }

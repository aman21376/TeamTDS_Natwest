# main.py

import argparse
import sys
import os
import logging
from detect_image import detect_image_deepfake
from detect_video import detect_video_deepfake
from detect_audio import detect_audio_deepfake
from config import LOGGING_LEVEL

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')

def analyze_file(media_type, file_path, output_dir):
    logging.info(f'Analyzing {media_type} file: {file_path}')
    if media_type == 'image':
        result = detect_image_deepfake(file_path)
    elif media_type == 'video':
        result = detect_video_deepfake(file_path)
    elif media_type == 'audio':
        result = detect_audio_deepfake(file_path)
    else:
        logging.error('Unsupported media type.')
        return
    confidence = result.get('confidence', 0.0)
    report = result.get('report', 'No report generated.')
    output_file = os.path.join(output_dir, os.path.basename(file_path) + '_report.txt')
    print(f"Deepfake Confidence Score: {confidence:.2f}")
    print(f"Report:\n{report}")
    try:
        with open(output_file, 'w') as f:
            f.write(f"Deepfake Confidence Score: {confidence:.2f}\n")
            f.write(f"Report:\n{report}\n")
        logging.info(f'Report saved to {output_file}')
    except Exception as e:
        logging.error(f'Error saving report: {e}')

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Tool')
    parser.add_argument('--type', choices=['image', 'video', 'audio'], required=True, help='Type of media to analyze')
    parser.add_argument('--path', required=True, help='Path to the media file or directory')
    parser.add_argument('--output', help='Directory to save the output reports', default='reports')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--recursive', action='store_true', help='Recursively process directories')
    args = parser.parse_args()
    setup_logging(args.verbose)
    logging.info('Deepfake Detection Tool Started')
    if not os.path.exists(args.path):
        logging.error(f'File or directory {args.path} does not exist.')
        sys.exit(1)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logging.info(f'Created output directory: {args.output}')
    if os.path.isfile(args.path):
        analyze_file(args.type, args.path, args.output)
    elif os.path.isdir(args.path):
        for root, dirs, files in os.walk(args.path):
            for file in files:
                file_path = os.path.join(root, file)
                analyze_file(args.type, file_path, args.output)
            if not args.recursive:
                break
    else:
        logging.error('Invalid path provided.')
        sys.exit(1)
    logging.info('Deepfake Detection Tool Finished')

if __name__ == '__main__':
    main()

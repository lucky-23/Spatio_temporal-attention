#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:03:14 2023

@author: kanchan
"""


import cv2
import torch
import torchvision.transforms as transforms
import time
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define transformations for preprocessing video frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Open a video stream

video_capture = cv2.VideoCapture('makingpasta.mp4')
original_frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

ground_truth_labels = []
with open("/makingpasta_gt.txt", 'r') as file:
    for line in file:
        start, end, label, *_ = line.strip().split()
        start, end = float(start), float(end)
        ground_truth_labels.append((start, end, label))

window_size = 1  # Number of frames in the sliding window
window_buffer = []  # Buffer to store frames in the sliding window
frame_count = 0  # Keep track of the current frame
video_duration = 186
bar_color = (0, 0, 0)  # Color of the bar (in BGR format)
bar_height = 20  # Height of the bar in pixels
bar_thickness = 5  # Thickness of the bar in pixels
fps = 30  # Adjust this value for the desired FPS
frame_delay = int(1000 / fps)

start_time = time.time()
display_width = 1280
display_height = 720

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('/pasta_output_video_new.avi', fourcc, fps, (display_width, display_height))

# Initialize the index of the active predicted segment
active_predicted_index = 0
active_predicted_start = 0  # Initialize active_predicted_start
active_predicted_end = 0  # Initialize active_predicted_end
elapsed_time =0
start_time = None
# Create a list to store displayed predicted labels
displayed_predicted_labels = []
displayed_gt_labels = []
gt_font_scale =0.8
predicted_font_scale = 0.8
ground_truth_progress = {label: 0.0 for _, _, label in ground_truth_labels}
while True:
    ret, frame = video_capture.read()
    frame_timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)/1000.0
    if start_time is None:
        start_time = frame_timestamp
    elapsed_time = frame_timestamp - start_time
    if not ret:
        break

    # Preprocess the frame
    frame_count = frame_count % video_duration
    bar_width = int((frame_count / video_duration) * frame.shape[1])
    input_tensor = transform(frame).unsqueeze(0).to(device)

    predicted_label = []

    with open("/makingpasta_prediction.txt", 'r') as file:
        for line in file:
            start, end, label = line.strip().split()
            start, end = float(start), float(end)
            predicted_label.append((start, end, label))

    # Update the sliding window buffer
    window_buffer.append((frame, predicted_label))

    # Ensure the sliding window is of the specified size
    if len(window_buffer) > window_size:
        window_buffer.pop(0)  # Remove the oldest frame

    overlay = np.zeros(frame.shape, dtype=np.uint8)
    for segment in ground_truth_labels:
        start, end, label = segment
        if start <= elapsed_time <= end:
            # Calculate the position and length of the bar within the ground-truth segment
            gt_start = start
            gt_end = end
            gt_segment_length = gt_end - gt_start

            if gt_start <= frame_timestamp <= gt_end:
                # Update the progress for the active ground-truth segment
                ground_truth_progress[label] = (frame_timestamp - gt_start) / gt_segment_length

            # Draw the ground-truth bar based on the progress
            bar_position = int((gt_start / video_duration) * frame.shape[0])
            bar_length = int(ground_truth_progress[label] * frame.shape[0])
#          
    for segment in predicted_label:
        start, end, label = segment
#        print("11111",start)
#        print("2222",end)
        if start <= elapsed_time <= end:
            # Calculate the position and length of the bar within the active predicted segment
            predicted_start = start
            predicted_end = end
            predicted_segment_length = predicted_end - predicted_start
#            print("start",predicted_start)
#            print("end",predicted_end)
            if predicted_start <= frame_timestamp <= predicted_end:
#               
                progress = (frame_timestamp - predicted_start) / predicted_segment_length
#                print("progress",progress)
                bar_position = int((predicted_start / video_duration) * frame.shape[0])
                bar_length = int(progress * frame.shape[0])
#                print("start",predicted_start)
#                print("progress",progress)
                # Display the label within the active predicted segment
#                cv2.putText(frame, f"Predicted: {label} ({start:.2f} - {end:.2f})",
#                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw the moving bar within the active predicted segment
                cv2.rectangle(frame, (bar_position, frame.shape[0] - bar_height - 20),
                              (bar_position + bar_length, frame.shape[0] - 20), (0, 0, 255), -1)
                 # Display the label at the end of the predicted timestamp bar
            
                if progress == 0.9939393939393951 or progress >=1 :
#                    print("true")
                    label_text = f"Predicted: {label} ({start:.2f} - {end:.2f})"
#                    print("label_text",label_text)
                    text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_x = bar_position + bar_length - text_size[0]
                    text_y = frame.shape[0] - 10
#                    cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                     # Add the displayed label to the list
                    displayed_predicted_labels.append(label_text)
              
                    
#    
 

    # ...
    # ...
    for i, segment in enumerate(ground_truth_labels):
        gt_label_text = f"Ground Truth: {segment[2]} ({segment[0]:.2f} - {segment[1]:.2f})"
        gt_label_x = 10
        gt_label_y = 30 + i * 30  # Adjust the vertical spacing between labels
        cv2.putText(frame, gt_label_text, (gt_label_x, gt_label_y), cv2.FONT_HERSHEY_SIMPLEX, gt_font_scale, (0, 255, 0), 2)

    for i, predicted_label_text in enumerate(displayed_predicted_labels):
        text_size, _ = cv2.getTextSize(predicted_label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        predicted_label_x = frame.shape[1] - text_size[0] - 10
        predicted_label_y = 30 + i * 30  # Adjust the vertical spacing between labels
        cv2.putText(frame, predicted_label_text, (predicted_label_x, predicted_label_y), cv2.FONT_HERSHEY_SIMPLEX, predicted_font_scale, (0, 0, 255), 2)
      
    timestamp_text = f"Time: {elapsed_time:.2f}"
    cv2.putText(frame, timestamp_text, (10, frame.shape[0] - bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    output_video.write(frame)
    frame_count += 1

    key = cv2.waitKey(frame_delay) & 0xFF
    if key == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()

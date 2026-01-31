#!/bin/bash

set -e  # Exit on error

echo "Creating storage directory..."
mkdir -p techtrack/storage

###############################################
# YOLO Model 1
###############################################
echo " Downloading YOLO model 1..."
wget --no-check-certificate \
  "https://www.dropbox.com/scl/fo/i4fdjtg9tigyjchjs61n0/AA2AaIy4HsoLHRkwmvVC5iU?rlkey=ke6ji9wfweorx0bd5phs2dlho&st=ixsk44d1&dl=1" \
  -O model_archive_1.zip

echo " Extracting YOLO model 1..."
mkdir -p techtrack/storage/yolo_model_1
unzip -q model_archive_1.zip -d techtrack/storage/yolo_model_1 || true
rm model_archive_1.zip

###############################################
# YOLO Model 2
###############################################
echo " Downloading YOLO model 2..."
wget --no-check-certificate \
  "https://www.dropbox.com/scl/fo/mpqudwxmwnm2lbf8d2mo7/AOftARd-DHMPcCp800cUl3g?rlkey=3fhqixu6lalcku9o259io6b5z&st=1iiljn0i&dl=1" \
  -O model_archive_2.zip

echo "Extracting YOLO model 2..."
mkdir -p techtrack/storage/yolo_model_2
unzip -q model_archive_2.zip -d techtrack/storage/yolo_model_2 || true
rm model_archive_2.zip

###############################################
# Test Videos
###############################################
echo "Downloading test videos..."
wget --no-check-certificate \
  "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/jpulido6_jh_edu/IQBJIuTDpeqiRZJvYQJxNREKAXLhjXybZNm2AKkQxC8vy6o?e=dU32hL=1" \
  -O test_videos.zip

echo "Extracting test videos..."
mkdir -p techtrack/storage/test_videos
unzip -q test_videos.zip -d techtrack/storage/test_videos || true
rm test_videos.zip

###############################################
# Logistics
###############################################
echo " Downloading logistics..."
wget --no-check-certificate \
  "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/jpulido6_jh_edu/IQC83xUcRiY3T5jOXuZQK__yAUXV9vahEIRIxBY8i0NKSEE?e=VqVzqz=1" \
  -O logistics.zip

echo "Extracting logistics..."
mkdir -p techtrack/storage/logistics
unzip -q logistics.zip -d techtrack/storage/logistics || true
rm logistics.zip

echo " All downloads and extractions completed successfully."




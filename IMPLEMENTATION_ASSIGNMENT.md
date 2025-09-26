# TechTrack Implementation Assignment

## Prerequisites

Before starting, make sure you have completed the following:

1. **Review lectures** on Object Detection, focusing on:

   * Inference Service (preprocessing, object detection, and non-maximum suppression)
   * Metrics and object detection *Mean Average Precision (mAP)*
   * Rectification Service (hard negative mining)
   * Deployment with Docker

2. **Study tutorials and resources**:

   * [OpenCV YOLO Tutorial](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html)
   * Skeleton code provided with docstrings for each function.

3. **Prepare datasets and resources**:

   * `techtrack/storage/logistics`: unzipped logistics dataset
   * `techtrack/storage/test_videos`: unzipped test videos
   * `techtrack/storage/yolo_models`: unzipped `yolo_model_1` and `yolo_model_2`
   * Do **NOT** commit these datasets into your GitHub repository.

4. **Install FFmpeg** and verify UDP streaming:

   * Terminal 1: Display video stream

     ```bash
     ffplay udp:127.0.0.1:23000
     ```
   * Terminal 2: Stream video file

     ```bash
     ffmpeg -re -i ./test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
     ```
   * Explanation: `ffplay` receives and displays the UDP stream at port 23000. `ffmpeg` streams the video at 30 fps. Ports can be adjusted if needed.

## Required Libraries

* Standard libraries (`os`, `sys`, `math`, `itertools`, etc.)
* `opencv-python` *(use `opencv-python-headless` for Docker)*
* `numpy`
* `matplotlib` / `seaborn`
* `pandas`

> **Note:** The updated `requirements.txt` includes all required packages. Replace `opencv-python` with `opencv-python-headless` for Docker deployments.

## Objectives

By the end of this assignment you will:

* Implement and extend the **Inference Service**
* Implement the **Rectification Service** (hard negative mining)
* Implement **Mean Average Precision (mAP)** evaluation
* Package the Inference Service as a **Dockerized container**

## Instructions

Use (your provisioned repository)[TODO: Insert link here] to fork the TechTrack base repository for this assignment into your personal GitHub account. Please update your current repository as there may be updates to the repository. Make your changes as directed by the instructions and push your changes into your repository. *Unit test automatically runs when you push new commits to your repository.*

### Part 1: Inference Service

1. **`model.py`**

   * Implement `predict()` to output **all YOLO predictions**.
   * Implement `post_process()` to filter predictions using `score_threshold`.

2. **`nms.py`**

   * Implement `filter()` to apply **Non-Maximum Suppression (NMS)** using **NumPy**.

3. **`metrics.py`**

   * Implement `evaluate_detections()` to evaluate predictions against ground truth.
   * Implement `calculate_precision_recall_curve()`.
   * The helper `calculate_map_x_point_interpolated()` is provided.
   * Complete `calculate_iou()` if not already done.

### Part 2: Rectification Service

4. **`loss.py`**

   * Implement `compute()` to calculate YOLO loss components.

5. **`hard_negative_mining.py`**

   * Implement `sample_hard_negatives()` to return top-N negative samples.

### Part 3: Preprocessing & Deployment

6. **`preprocessing.py`**

   * Implement `capture_video()` to yield every *drop-rate*’th frame using `yield`.
   * Ensure efficient frame streaming for live detection.

7. **`app.py`**

   * Implement `run()` to integrate all modules:

     * Capture stream via UDP (preprocess)
     * Detect objects in frames (model)
     * Apply NMS (nms)
     * Print per-frame detections (bounding box, class\_id, score)
     * Save frames with detections in `storage/detections/`
   * Do not commit the contents of `storage/detections/`.

8. **`README.md`**

   * Write a **Quick Start guide** showing how to:

     * Build and run the Dockerized Inference Service
     * Stream video with FFmpeg
     * Use `curl` or other tools to test endpoints


## Submission & Evaluation

Check in (i.e., git push) all implementation files into your provisioned GitHub repository. Provide the **repository URL** before the deadline to receive credit. After checking this in, GitHub Classroom with automatically run the autograder. 

> NOTE: Do not change `.yaml` files and `test/*` files! This will flag your repository and you will not recieve credit for your assignment.

1. **Push your work**  
   Check in (i.e., `git push`) all implementation files into your provisioned GitHub repository.  

2. **Submit your repository URL**  
Provide the **repository URL** before the deadline to receive credit. 

3. **Autograder execution**  
Once you push to GitHub, **GitHub Classroom** will automatically run the autograder on your submission.  
- You can view the results under the **Actions** tab of your repository.  
- Each push to your repository will re-trigger the autograder.  
- The autograder runs unit tests to check correctness of your code and may also check for:
  - File naming conventions  
  - Method signatures  
  - Correctness of outputs  
  - Presence of required files

4. **Checking your grade**  
- Go to your repository on GitHub.  
- Click the **Actions** tab.  
- Select the latest workflow run (triggered by your most recent commit).  
- Expand the **Autograder job** to see detailed test results.  
- A ✅ indicates a passed test; a ❌ indicates a failed test.  

5. **Resubmissions**  
- If you fail tests, you can fix your code and push again.  
- Each new commit will re-run the autograder.  
- Only the **latest successful run before the deadline** counts toward your grade.  

---

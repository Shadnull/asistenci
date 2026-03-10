
# Automated Facial Recognition Attendance System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Academic Context](#academic-context)
3. [System Architecture](#system-architecture)
5. [Installation & Prerequisites](#installation--prerequisites)

## Project Overview

This repository contains an automated attendance logging system based on Computer Vision and Deep Learning techniques. The system detects and extracts human faces from digital images, computes a 128-dimension facial encoding vector, and calculates the Euclidean distance against a pre-registered database to determine identity and log attendance.

## Academic Context

This project is developed as part of the academic curriculum for the **Computer Vision** course.

* **Academic Group:** IA-51
* **Subject:** Computer Vision (Visión por Computadora)
* **Execution Status:** 100% Positive Attendance Registered.

### Registered Cohort

| ID | Full Name | Attendance Status |
|---|---|---|
| 01 | Donnovan Joel Creano Rodriguez | Positive |
| 02 | Rodrigo Ortega Andrade | Positive |
| 03 | Bernal Loma José Angel | Positive |
| 04 | Arce Armenta Fernando De Jesús | Positive |
| 05 | Ulises de jesus gongora pacheco | Positive |
| 06 | Diego Alejandro Durán Tapia | Positive |
| 07 | Johan Fernando sierra López | Positive |

## System Architecture

The facial recognition pipeline operates under the following sequence:

1. **Image Acquisition & Preprocessing:** The system reads the target image using OpenCV and converts the color space from BGR (OpenCV default) to RGB (standard for `face_recognition`).
2. **Face Detection:** Utilizes Histogram of Oriented Gradients (HOG) or Convolutional Neural Networks (CNN) to detect bounding boxes around human faces in the frame.
3. **Feature Extraction:** Projects the detected facial landmarks into a 128-dimensional embedding space.
4. **Distance Calculation:** Compares the generated encodings with known encodings using a linear Support Vector Machine (SVM) or simple thresholded Euclidean distance calculations to find the closest algorithmic match.
5. **Annotation & Output:** Overlays bounding boxes and identity labels over the processed image array.



## Installation & Prerequisites

To deploy this environment, specific C++ build tools are required prior to installing Python dependencies due to the `dlib` backend requirement.

### 1. System Dependencies

* **Ubuntu/Debian:** `sudo apt-get install cmake build-essential libopenblas-dev liblapack-dev`
* **macOS:** `brew install cmake`
* **Windows:** Install Visual Studio Community with "Desktop development with C++" workload and CMake.

### 2. Python Environment Setup

It is highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -U pip
pip install opencv-python numpy face_recognition

```

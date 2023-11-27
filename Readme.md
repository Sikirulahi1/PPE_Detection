Certainly! Here's a more detailed README file that you can use for your Personal Protective Equipment (PPE) detection project:

---

# Personal Protective Equipment (PPE) Detection

## Overview

This project aims to detect personal protective equipment (PPE) components, including facemasks, safety vests, and hard hat helmets. The detection model is trained using a custom dataset obtained from [Roboflow](https://roboflow.com/) and implemented with YOLOv8l.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Introduction

In industrial and safety-conscious environments, ensuring that individuals wear proper personal protective equipment is crucial. This project addresses the need for automated detection of key PPE components through a robust and efficient algorithm.

## Requirements

To use this PPE detection system, ensure that you have the following requirements:

### Hardware:

- GPU with CUDA support (recommended for faster inference)
- Minimum of 8GB RAM

### Software:

- Python 3.7 and above
- Required dependencies (specified in `requirements.txt`)

## Installation

Follow these steps to set up the project on your local machine:

```bash
# Clone the repository
git clone https://github.com/Sikirulahi1/PPE_Detection.git

# Navigate to the project directory
cd PPE_Detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

To utilize the PPE detection algorithm on your own images or videos, follow these commands:

- Just change the video path to your own path to the video or image directory.
- run PPEDetection.py

## Training

I trained the PPE detection model on a custom dataset using the following steps:

1. I downloaded the dataset from [Roboflow](https://roboflow.com/).
2. I trained the model using YOLOv8.


## Results

View the performance metrics and sample results obtained from the trained model in the `results` directory.

## Acknowledgments

I would like to express my gratitude to the following:

- [Roboflow](https://roboflow.com/) for providing the PPE dataset.
- YOLOv8 for the powerful object detection framework.


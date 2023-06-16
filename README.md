# Neural-Network-Final-Project
## This is my 8th semester Neural Network final project.


# Face Mask Detection Using Convolutional Neural Networks (CNN)

This project aims to develop a Convolutional Neural Network (CNN) model for accurately detecting whether individuals in an image or video stream are wearing face masks or not. The project utilizes the power of deep learning and computer vision techniques to contribute to public health and safety, especially during the COVID-19 pandemic.

## Table of Contents
- [Introduction](#introduction)
- [Related Works](#related-works)
- [Approach](#approach)
- [Experiments](#experiments)
- [Results](#results)
- [Analysis](#analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [NB](#NB)

## Introduction
Face mask detection has become essential in maintaining public health and safety, particularly in the context of the COVID-19 pandemic. This project utilizes Convolutional Neural Networks (CNNs), which have proven to be powerful tools in image classification tasks, making them suitable for face mask detection. The goal is to develop a CNN model capable of accurately classifying whether a person is wearing a face mask or not.

## Related Works
- "Real-time face mask detection in video streams with deep learning" by Amarjot Singh et al. (2020)
- "Face Mask Detection using Convolutional Neural Networks with MobileNetV2" by Renuka Mohanraj et al. (2021)
- "A Study on Face Mask Detection using Convolutional Neural Networks" by S. Sakthi Vinayakam et al. (2020)

## Approach
The project follows the standard approach for face mask detection using CNNs. The CNN architecture is designed specifically for image and spatial data processing, consisting of convolutional layers for feature extraction, pooling layers for downsampling, and fully connected layers for decision-making. This hierarchical design allows the CNN to effectively capture complex patterns and relationships in the data.

## Experiments
The experiments involve the following steps:
1. Dataset preparation: Collect or obtain a labeled dataset containing images of individuals with and without face masks. Ensure a balanced distribution between the two classes.
2. Model architecture design: Design the CNN model architecture, including convolutional layers, pooling layers, and fully connected layers.
3. Training: Train the model using the prepared dataset, specifying the appropriate loss function, optimization algorithm, and evaluation metrics.
4. Evaluation: Evaluate the trained model using metrics such as accuracy, confusion matrix, precision, recall, and F1-score.

## Results
The results of the experiments indicate the effectiveness of CNNs for face mask detection. The trained CNN model achieves high accuracy in classifying whether individuals are wearing face masks or not. The success rate demonstrates the potential of the model for real-world applications, contributing to public health and safety efforts.


Certainly! Here's an updated version of the README with three pictures added to the Results section:

## Results
The results of the experiments indicate the effectiveness of CNNs for face mask detection. The trained CNN model achieves high accuracy in classifying whether individuals are wearing face masks or not. The success rate demonstrates the potential of the model for real-world applications, contributing to public health and safety efforts.

Here are some visual examples of the model's performance:

|No mask detected|No mask detected|Mask detected|
|:-:|:-:|:-:|
![No mask detected](https://github.com/Istiaq66/Neural-Network-Final-Project/blob/master/test1.jpeg) | ![No mask detected](https://github.com/Istiaq66/Neural-Network-Final-Project/blob/master/test2.jpeg) | ![Mask detected](https://github.com/Istiaq66/Neural-Network-Final-Project/blob/master/test3.jpeg)


## Analysis
The analysis of the project includes considerations for false positives and false negatives in face mask detection. While the model achieves high accuracy, it is essential to understand the implications of misclassifications. The quality and diversity of the training dataset also play a vital role in the model's performance. Further optimization possibilities are suggested, such as fine-tuning the model architecture, exploring advanced CNN architectures, adjusting hyperparameters, and incorporating techniques like data augmentation.

## Installation
To run this project locally, follow these steps:
1. Clone the repository: `git clone https://github.com/Istiaq66/Neural-Network-Final-Project.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage
To use the face mask detection model, you need to follow these steps:
1. Load the pre-trained model.


2. Preprocess the input data, such as resizing images or extracting frames from videos.
3. Feed the processed data into the model for prediction.
4. Interpret the model's output to determine whether individuals are wearing face masks or not.

For a detailed example, refer to the [notebook](https://colab.research.google.com/drive/1YUJYDlZeEyMf_f2Ts5P9QIk_Dbp-Iftj#scrollTo=0EtGjlM3jf20) provided in this repository.

## Contributing
Contributions to this project are welcome. If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request, explaining your changes and their benefits.

## NB
You are free to modify and distribute the codebase, as long as you include the original license file and attribute the authors.

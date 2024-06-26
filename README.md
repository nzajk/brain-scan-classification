# Brain Scan Classification

This repository contains a brain scan classifier that utilizes transfer learning with EfficientNetB0 architecture to classify images into four categories: no tumor, meningioma, glioma, or pituitary. The classifier is designed to assist medical professionals in diagnosing brain scans efficiently and accurately.

## Overview

Brain scans, such as MRI or CT images, are crucial in diagnosing various brain conditions, including tumors. Automating the process of analyzing these scans can significantly aid healthcare professionals in making timely and accurate diagnoses. This classifier leverages the power of deep learning and transfer learning to achieve this goal.

### Model Architecture

The classifier employs the EfficientNetB0 architecture, a state-of-the-art convolutional neural network (CNN) known for its efficiency and effectiveness in image classification tasks. Transfer learning is utilized by fine-tuning the pre-trained EfficientNetB0 model on a dataset of brain scan images.

### Dataset

The classifier is trained on a dataset consisting of labeled brain scan images categorized into four classes: no tumor, meningioma, glioma, and pituitary. The dataset is curated to ensure diversity and representativeness across all classes, enabling the model to generalize well to unseen data.

### Training

The model is trained using TensorFlow and Keras, making use of efficient data pipelines for loading and preprocessing images. During training, the weights of the pre-trained EfficientNetB0 model are fine-tuned on the brain scan dataset using techniques such as gradient descent optimization and learning rate scheduling to achieve optimal performance.

### Model Evaluation

The performance of the classifier is assessed using standard metrics such as accuracy and loss. These metrics provide insights into the model's ability to correctly classify brain scans into their respective categories.

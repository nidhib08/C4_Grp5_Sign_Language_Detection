# C4_Grp5_Sign_Language_Detection
# Real-Time Sign Detection using TensorFlow and CNN

## Abstract

Sign language recognition systems provide essential accessibility for people with hearing impairments by bridging communication gaps. In this work, we develop a real-time sign phrase detection system using a convolutional neural network (CNN) in TensorFlow. Our system captures live video frames from a camera, preprocesses these frames (resizing, normalization), and classifies them into one of five predefined sign phrases. We built a small custom dataset (20 images per phrase, total 100 images), split into an 80/20 train-test configuration. The trained CNN model operates on 400×400 RGB images. We integrate the model inference into a continuous video feed using OpenCV, enabling real-time detection. We also analyze challenges such as limited data, intra-class variability, and generalization. Finally, we discuss directions for improvement including data augmentation, model optimization, and possible integration with hand-landmark-based methods.

**Keywords —** sign language recognition; convolutional neural networks; real-time inference; TensorFlow; computer vision

---

## I. Introduction

Communication through sign language is a vital modality for individuals who are deaf or hard of hearing. However, many non-signers struggle to interpret sign language, which can pose a barrier in everyday interactions. Automated sign language recognition (SLR) systems hold promise in bridging this gap by translating gestures into text or speech. Existing systems often leverage large datasets, depth cameras, or motion-based architectures.

In this paper, we present a lightweight, real-time sign detection system that uses a standard RGB camera and a CNN built in TensorFlow. We focus on static sign phrases (rather than continuous sequences), with a vocabulary of five common phrases. Despite limited data, our system demonstrates feasibility for practical deployment.

Our main contributions are:

1. Building a CNN-based classifier for five sign phrases using a small, custom dataset.
2. Deploying the model in real-time via live video feed using OpenCV.
3. Identifying and discussing the challenges in small-dataset training, model generalization, and real-world deployment.

The rest of the paper is organized as follows: Section II surveys related work; Section III describes our methodology; Section IV presents discussion and analysis; Section V outlines future work; and Section VI concludes.

---

## II. Literature Review

Research in sign language recognition has seen rapid advancement due to deep learning. We review key works relevant to static gesture detection, real-time systems, and model architectures.

### A. CNN-based Static Gesture Recognition

Kothadiya et al. proposed *Deepsign*, a deep learning model combining LSTM and GRU for recognizing Indian Sign Language (ISL) from video frames. Their system achieved about 97% accuracy on an 11-sign dataset.
In another study, Kang et al. used depth camera data with a CNN to perform real-time fingerspelling recognition.
Deshmukh (2024) designed a CNN-based real-time recognition system for Indian Sign Language to improve accessibility and explored model deployment for mobile using TensorFlow Lite.

### B. Real-Time Detection Models

Alaftekin et al. (2024) used a YOLOv4-based object detection algorithm (with CSPNet) for real-time recognition of Turkish Sign Language, showing that object-detection models can achieve high speed and accuracy.
Hoque et al. (2018) applied a Faster R-CNN approach to Bangladeshi Sign Language (BdSL), detecting signs in real-time from video frames.

### C. Hybrid and Landmark-Based Approaches

Verma et al. combined MediaPipe hand landmark detection with CNNs to boost real-time recognition efficiency and accuracy.
For recognition of continuous or dynamic sign sequences, models combining spatial networks (e.g., ResNet) with temporal modules (e.g., LSTM) have been proposed. For instance, a ResNet + LSTM architecture achieved high performance on Argentine Sign Language video sequences.

### D. Summary of Gaps

* Many systems rely on large datasets or complex architectures; small-data solutions are less explored.
* Real-time systems often use detection frameworks (YOLO, R-CNN) rather than simple classifiers.
* Hybrid approaches combining landmark detection and CNNs have shown strong performance but increase system complexity.
* Lightweight but accurate real-time solutions for phrase-level (not just alphabet-level) recognition are still limited.

These observations motivate our work: building a minimal, efficient, real-time phrase recognizer using a small dataset and a pure CNN.

---

## III. Methodology

This section details our dataset, preprocessing, model design, and inference pipeline.

### A. Dataset Collection

We created a custom dataset comprising five static sign phrases: “Hello,” “I Love You,” “Yes,” “No,” and “Thank You.” For each phrase, 20 images were captured using a standard webcam under indoor lighting conditions. This yields 100 total samples. To introduce variability, we captured signs from different hand orientations and slightly varied backgrounds. The dataset is split into 80% for training (16 images per class) and 20% for testing (4 images per class).

### B. Preprocessing

Each image undergoes the following pipeline:

1. **Resizing** – each image is resized to 400 × 400 pixels.
2. **Color Conversion** – images are converted from BGR to RGB color space.
3. **Normalization** – pixel values are converted to float32 and scaled between 0 and 1.
4. **Batch Dimension** – for inference, the single image is expanded to form a batch of size one, shape (1, 400, 400, 3).

### C. Model Architecture

We implement an 8-layer Convolutional Neural Network (CNN) in TensorFlow/Keras. The architecture is roughly:

* Input: (400, 400, 3)
* Multiple convolutional layers with ReLU activations and max-pooling layers
* Fully-connected (dense) layers with dropout for regularization
* Softmax output layer with five units

Training is performed with categorical cross-entropy loss and the Adam optimizer. Given the small dataset, techniques like early stopping and dropout are used to reduce overfitting.

### D. Real-Time Inference Pipeline

For deployment, we use OpenCV to read frames from a live camera feed. The inference loop performs:

1. Capture frame
2. Resize and normalize frame
3. Run inference via `model.predict()`
4. Obtain predicted class via softmax argmax
5. Overlay predicted phrase and confidence on the frame

This feedback loop runs continuously to support real-time detection.

### E. System Tools

The implementation uses:

* TensorFlow / Keras – for model building and training
* OpenCV – for image capture and display
* NumPy – for array operations

MediaPipe and TensorFlow Lite can be used in future versions to enhance detection and optimize inference speed.

---

## IV. Results and Discussion

This section presents qualitative analysis and discussion of system performance.

### A. Feasibility and Real-Time Performance

Given that our CNN processes 400×400 RGB images, inference speed can be fast on most modern hardware. The lightweight architecture and small input size support real-time performance, typically achieving frame rates of 10–30 fps depending on system hardware.

### B. Challenges of Limited Data

A dataset of only 100 images (20 per class) is small for deep learning. Overfitting remains a major challenge, as the model may memorize background or lighting patterns. Data augmentation techniques are essential to improve generalization.

### C. Intra-Class Variability

Signs performed by different individuals may vary in position and angle. Limited diversity in training data can lead to poor generalization to unseen users. This issue highlights the need for a larger, more varied dataset.

### D. Absence of Temporal Modeling

Our system currently handles only static poses, not dynamic gestures. Many real-world sign phrases involve motion and hand trajectory. Incorporating temporal modeling would allow recognition of complex gestures.

### E. Generalization Across Users

Since training data were collected from a small number of individuals, user-dependent bias is likely. Real-world deployment will require collecting data from multiple signers under different lighting conditions to improve robustness.

### F. Comparison with Related Approaches

Compared to object-detection-based systems such as YOLOv4, our approach is simpler and computationally lighter but more sensitive to background noise. Landmark-based systems using MediaPipe can provide robustness to background changes but increase complexity. Architectures combining CNNs with LSTMs are more accurate for continuous recognition tasks but are computationally heavier.

---

## V. Future Work

1. **Data Augmentation and Expansion**

   * Introduce rotations, flips, and brightness adjustments to create synthetic data.
   * Collect additional samples from multiple signers and environments.

2. **Model Optimization**

   * Convert the model to TensorFlow Lite for deployment on edge devices.
   * Apply quantization to reduce model size and improve speed.

3. **Integration of Hand Landmark Detection**

   * Incorporate MediaPipe Hands to extract 21 hand landmarks and combine with CNN features for better accuracy.

4. **Temporal Modeling**

   * Extend to dynamic signs using CNN-LSTM or Transformer architectures.

5. **User Study and Testing**

   * Conduct evaluations with multiple users to measure performance and usability.

6. **Robustness Improvements**

   * Implement background segmentation or attention mechanisms to focus on hands.
   * Train with augmented or adversarial data to handle real-world variability.

---

## VI. Conclusion

This paper presented a real-time sign phrase detection system using TensorFlow and CNN. The model classifies five predefined phrases in real time from a live camera feed. Despite the small dataset, the approach demonstrates feasibility for real-world applications. Key challenges include limited data, user variability, and lack of temporal modeling. With further dataset expansion and model optimization, the system can evolve into a scalable and effective assistive technology for communication between signers and non-signers.

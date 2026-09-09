# Physics-Informed CNN for Fatigue Crack Initiation Prediction

## Overview
Traditional data-driven neural networks often act as black boxes, learning patterns that violate physical laws. This project implements a **Physics-Informed Convolutional Neural Network (PI-CNN)** that embeds crystallographic misorientation physics and Schmid factor constraints directly into the loss function to predict fatigue crack initiation sites in polycrystalline metals (such as Ti-6Al-4V and AA7075).

## Key Features & Architecture
- **Dual-Branch Architecture**: Combines an Image Processing Branch (EBSD spatial features) with a Crystallographic Branch (Euler angles & Schmid factors).
- **Physics-Augmented Loss Function**:
  Augmented with grain boundary misorientation and Schmid factor constraints -
  L_PI = L_data + lambda_1(L_misorient) + lambda_2(L_Schmid)
- **Feature Fusion**: Uses an attention mechanism to combine crystallographic features with spatial images.

## Key Results
- **19% Accuracy Improvement** over standard data-driven CNN baselines.
- **94% Physical Consistency Rate** on predicted crack sites.
- **2.1x Reduction** in physical constraint violations compared to traditional models.

## Tech Stack
- Python, PyTorch / TensorFlow, NumPy, Matplotlib, Scikit-Learn

# Object Detection Model Training and Inference Pipeline

This document outlines the requirements, installation, and execution steps for training and running an object detection model.

## Hardware/Software Requirements

*   **GPU:** NVIDIA Tesla P100-PCIE-16GB (or a similar GPU with at least 16GB of VRAM)
*   **CUDA Version:** 12.4
*   **Python:** 3.8+
*   **Operating System:** Linux (recommended), macOS, or Windows (with appropriate CUDA setup)

## Dependency Installation

It is highly recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

1.  **Create and activate a virtual environment (optional but recommended):**
    *   Using `venv`:
        ```bash
        python3 -m venv venv
        source venv/bin/activate  # On Linux/macOS
        # venv\Scripts\activate  # On Windows
        ```
    *   Using `conda`:
        ```bash
        conda create -n myenv python=3.8
        conda activate myenv
        ```

2.  **Install the required packages:**
    Clone the repository (if you haven't already) and navigate to its root directory. Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Step-by-Step Execution Commands

The following scripts must be run in the specified order. Replace placeholder paths and parameters with your actual values.

### 1. Data Augmentation

This script preprocesses and augments your dataset.

```bash
python3 data_augmentation.py 



### 2. Model Training

This script trains the object detection model using the augmented dataset.

python3 train.py 

### 3. Evaluation Metrics

This script evaluates the trained model on a validation or test set.

python3 evaluation_metrics.py 

### 4. Inference

This script performs inference on new images or videos using the trained model.

python3 infer.py

## Validation Scores

The following Best AP50 (Average Precision at IoU threshold 0.50) scores were achieved during development:

| Class Name                                | Best AP50 Score |
| :---------------------------------------- | :-------------- |
| basketball field                          | 0.4248          |
| building                                  | 0.8451          |
| crosswalk                                 | 0.9200          |
| football field                            | 0.3911          |
| graveyard                                 | 0.6551          |
| large vehicle                             | 0.5961          |
| medium vehicle                            | 0.7461          |
| playground                                | 0.2678          |
| roundabout                                | 0.4261          |
| ship                                      | 0.7664          |
| small vehicle                             | 0.9062          |
| swimming pool                             | 0.6840          |
| tennis court                              | 0.5875          |
| train                                     | 0.4804          |
| **OVERALL (Hybrid mAP50, excl. 'small-object')** | **0.6212**      |
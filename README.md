# Review Summarization using GPT-2

This repository demonstrates how to fine-tune the GPT-2 model for summarizing Amazon Fine Food Reviews. The goal is to generate concise and coherent summaries for given reviews using the pre-trained GPT-2 model from Hugging Face.

---

## **Table of Contents**

- [Overview](#overview)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Implementation](#implementation)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation](#3-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Challenges and Future Work](#challenges-and-future-work)
- [References](#references)

---

## **Overview**

This project fine-tunes a pre-trained GPT-2 model to summarize product reviews from the Amazon Fine Food Reviews dataset. Summaries are evaluated using ROUGE metrics to assess the precision, recall, and F1-scores.

---

## **Dataset**

- **Source:** [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Columns Used:**
  - `Text`: The detailed review of the product.
  - `Summary`: A short summary of the review.

The dataset was cleaned and preprocessed to remove duplicates, null values, HTML tags, and other irrelevant characters.

---

## **Setup and Installation**

### Prerequisites:
1. Python 3.8+
2. PyTorch
3. Hugging Face Transformers library
4. Other dependencies: `pandas`, `sklearn`, `nltk`, `bs4`, `rouge`.

### Steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/NitishKumar1123/Review-Summarization-using-GPT2-of-Amazon-Fine-Food-Reviews-dataset.git
    cd Review-Summarization-using-GPT2-of-Amazon-Fine-Food-Reviews-dataset
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Download the dataset if not provided:
    ```bash
    kaggle datasets download -d snap/amazon-fine-food-reviews
    ```

4. Preprocess the dataset:
    Run the `data_preprocessing.py` script to clean and preprocess the dataset.

---

## **Implementation**

### **1. Data Preprocessing**
- Removed duplicates and null values.
- Applied preprocessing steps such as removing HTML tags, lowercasing, and stripping punctuation and numbers.
- Combined `Text` and `Summary` columns using "TL;DR" as a separator for training.
- Saved the preprocessed data as `Preprocessed_Reviews.pkl`.

### **2. Model Training**
- **Tokenizer & Model:** Hugging Face GPT-2 with additional EOS token handling.
- **Custom Dataset Class:** Prepared reviews for training with padding/truncation.
- **Training Configuration:**
  - Batch Size: 32
  - Learning Rate: 3e-4
  - Epochs: 1 (demonstration purposes)
- Model saved as `fine_tune_gpt2_model1` for inference.

### **3. Evaluation**
- Generated summaries using the fine-tuned model.
- Evaluated summaries using ROUGE metrics.

---

## **Results**

Sample Results:

| **Review Text**                                                                                   | **Actual Summary**           | **Generated Summary**      | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
|---------------------------------------------------------------------------------------------------|------------------------------|----------------------------|-------------|-------------|-------------|
| "i ordered three of these but they were not like the picture antlers were cut in half..."         | "not like the picture"       | "they werent cut in half" | P: 0.80     | P: 0.50     | P: 0.80     |
| "this is a good product to use if you train with a mix of treats i just break the treats..."     | "good product for training" | "good training"           | P: 0.50     | P: 0.00     | P: 0.50     |

---

## **Usage**

1. **Fine-tune the Model:**
   Run the training script to fine-tune GPT-2 on your dataset.
   ```bash
   python train_model.py
   ```

2. **Generate Summaries:**
   Use the `generate_summary.py` script to infer summaries for new reviews.
   ```bash
   python generate_summary.py --input "Review Text Here"
   ```

3. **Evaluate Model:**
   Compute ROUGE scores using the `evaluate.py` script:
   ```bash
   python evaluate.py
   ```

---

## **Challenges and Future Work**

### **Challenges:**
- Computational constraints limited the number of training epochs and batch size.
- Variability in ROUGE scores for shorter reviews.

### **Future Work:**
- Hyperparameter optimization for better performance.
- Evaluate additional metrics and include human evaluations.
- Implement data augmentation techniques.
- Experiment with other transformer models like T5 or BART.

---

## **References**

1. [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
2. [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
3. [Fine-Tuning GPT-2 for Beginners](https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners)

---

Developed by [Nitish Kumar](https://github.com/NitishKumar1123).


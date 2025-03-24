# Evaluating the Effectiveness of Machine Learning Techniques for Pneumonia Detection in Chest X-Ray Images

Author: Denali Carpenter

## Executive Summary

This project explores various machine learning and deep learning approaches to automatically diagnose pneumonia from chest X-ray images. By comparing traditional methods (Random Forests and SVM) with custom and transfer learning convolutional neural networks (including VGG16 and DenseNet121), we determined that fine-tuned pretrained CNNs—especially VGG16—provide the best performance, achieving nearly 95% accuracy. High recall and precision are critical in medical settings, and our best model minimizes false negatives, potentially supporting early diagnosis and improved patient outcomes.

## Rationale

Pneumonia remains a significant cause of morbidity and mortality worldwide. Rapid and accurate diagnosis is crucial for timely treatment, particularly in resource-limited settings. By automating the detection process with machine learning, healthcare professionals can benefit from quicker, more reliable screenings, reducing human error and ultimately saving lives.

## Research Question

What is the most effective and efficient model for predicting pneumonia diagnoses from chest X-ray images, and can we accurately distinguish between viral and bacterial pneumonia using these images?

## Data Sources

The primary dataset used is from Kaggle, originally sourced from the research by Kermany et al. (2018):
- Dataset: Chest X-Ray Images for Pneumonia Detection
- Source: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- Details: Nearly 6,000 chest X-ray images, divided into training, testing, and validation sets, with labels for NORMAL, Viral Pneumonia, and Bacterial Pneumonia.

## Methodology

The project workflow involves:

- Data Cleaning & Preprocessing:
	- Removing duplicates, converting images to a consistent grayscale format, and addressing class imbalances.
- Exploratory Data Analysis (EDA):
	- Analyzing image features such as brightness, contrast, texture (via Gray-Level Co-Occurrence Matrix), edge detection, and Fourier energy.
- Feature Engineering & Dimensionality Reduction:
	- Extracting statistical and texture-based features along with applying PCA.
- Modeling:
	- Training baseline models (Dummy Classifier, Random Forest, SVM) and advanced models (custom CNN, VGG16, DenseNet121).
	- Hyperparameter tuning and evaluation using accuracy, precision, recall, F1-score, and AUC-ROC.
- Model Evaluation & Comparison:
	- Comparing all models and identifying that a fine-tuned VGG16 model offers the best performance for pneumonia detection.

## Results

**Performance Metrics for Models**

| Model        | Accuracy | Precision | Recall  | F1 Score |
|--------------|----------|-----------|---------|----------|
| Dummy        | 0.6262   | 0.6262    | 1.0000  | 0.7701   |
| RF (Base)    | 0.6893   | 0.6690    | 0.9974  | 0.8152   |
| RF (Tuned)   | 0.6942   | 0.6727    | 0.9974  | 0.8037   |
| SVM (Base)   | 0.7039   | 0.6802    | 0.9948  | 0.8080   |
| SVM (Tuned)  | 0.6958   | 0.6743    | 0.9948  | 0.8038   |
| Ensemble RF  | 0.6942   | 0.6725    | 0.9974  | 0.8030   |
| CNN (Base)   | 0.8835   | 0.8889    | 0.9302  | 0.9091   |
| VGG16        | 0.9482   | 0.9361    | 0.9845  | 0.9597   |
| DenseNet121  | 0.8786   | 0.8578    | 0.9664  | 0.9089   |

- Baseline Models:
	- Traditional methods (Random Forest and SVM) achieved accuracies around 69–70%.
- Deep Learning Models:
	- A custom CNN achieved an accuracy of about 88%.
	- The fine-tuned VGG16 model delivered nearly 95% accuracy with high recall (over 98%) and precision, indicating its potential for reliable clinical screening.
         -  ![alt text](images/VGG16_ROC.png 'VGG16 ROC Testing Set')
- Interpretation:
	- The high recall of the best model minimizes false negatives—a critical attribute for medical diagnostics.
	- Feature importance analysis further validated the relevance of texture and frequency-based features in distinguishing pneumonia cases.

## Next Steps
- Data Enrichment and Balance:
	- Expand the dataset to include more balanced examples, especially for viral pneumonia, to improve model generalization.
- Model Fine-Tuning and Ensemble Strategies:
	- Further fine-tune the CNN models (e.g., by unfreezing more layers) and explore ensemble methods combining various CNN architectures.
- Enhance Interpretability:
	- Incorporate techniques like Grad-CAM or saliency maps to visualize model attention on X-ray images and build trust among clinicians.
- Clinical Integration:
	- Validate the model prospectively in clinical settings and integrate it into existing diagnostic workflows to support decision-making.

## Outline of Project
- Notebook 1: [Exploratory Data Analysis (EDA) and Baseline Model](https://github.com/DenaliCarpenter/Predicting-Pneumonia-from-Chest-X-rays/blob/main/EDA%20and%20Initial%20Model.ipynb)
- Notebook 2: [CNN Modeling](https://github.com/DenaliCarpenter/Predicting-Pneumonia-from-Chest-X-rays/blob/main/Final%20CNN%20Modeling.ipynb)
- Notebook 3: [Final Models for Pneumonia Detection in X-Rays¶](https://github.com/DenaliCarpenter/Predicting-Pneumonia-from-Chest-X-rays/blob/main/Final%20Modeling%20Steps.ipynb)

## Contact and Further Information

For additional details, questions, or collaboration inquiries, please contact denalicarpenter@gmail.com.
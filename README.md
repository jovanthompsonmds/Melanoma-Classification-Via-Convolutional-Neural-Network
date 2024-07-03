# Melanoma Classification Via Convolutional Neural Network  

## Overview  
This project leverages a Convolutional Neural Network (CNN) to classify melanoma skin lesions as benign or malignant, aiming to enhance early detection accuracy. The model was trained and evaluated on a dataset of 13,900 high-resolution images, achieving 91.20% accuracy and an AUC-ROC score of 0.97. By using TensorFlow, Keras, and Google Cloud Platform (GCP), the project delivers a scalable and reliable diagnostic tool for healthcare professionals and individuals.  

## Features  
- **Data Preprocessing**: Resizing and normalizing images for consistent input to the CNN.  
- **CNN Architecture**: Four convolutional layers with increasing filters, pooling layers, and dense layers for classification.  
- **Performance Evaluation**: Confusion matrix, accuracy, and ROC-AUC curve to assess model reliability.  
- **Cloud Deployment**: Model deployed on GCP for scalability and accessibility.  

## Files  
1. `Melanoma Classification Via Convolutional Neural Network.ipynb`  
   - Code for data preprocessing, CNN model training, evaluation, and visualization.  
2. `melanoma_cnn_model.h5`  
   - Trained CNN model for deployment and predictions.  
3. `Melanoma Cancer Image Dataset`  
   - Image dataset used for training and testing.  
4. `Melanoma Classification Via Convolutional Neural Network - Google_Cloud_Deployment.py`  
   - Python script to deploy the model on GCP for real-time predictions.  

## Requirements  
- Python 3.8 or higher  
- Required Python libraries:  
  - `tensorflow`  
  - `numpy`  
  - `matplotlib`  
  - `scikit-learn`  
  - `opencv-python`  

Install the dependencies using:   
pip install tensorflow numpy matplotlib scikit-learn opencv-python  

## Usage
1. Clone the repository:
git clone https://github.com/jovanthompsonmds/Melanoma-Classification-Via-Convolutional-Neural-Network.git

2. Open the Jupyter Notebook and run the code to:
- Preprocess the images.
- Train the CNN model.
- Evaluate performance metrics.
- Visualize results (e.g., accuracy, confusion matrix, ROC-AUC curve).

3. Deploy the trained model on GCP for real-time predictions using the Melanoma Classification Via Convolutional Neural Network - Google_Cloud_Deployment.py script.

## Insights and Conclusions
The CNN model demonstrated high performance, significantly outperforming existing smartphone apps for melanoma detection. The confusion matrix showed a balanced classification of benign and malignant lesions, and the ROC-AUC score of 0.97 indicates excellent discrimination capability. This project provides a scalable and reliable tool for early melanoma detection, aiding healthcare professionals and patients in timely diagnosis and treatment.

### 1. Key Insights from the CNN Model for Melanoma Classification

1. **Objective & Problem Statement**
   - The project aims to develop a **Convolutional Neural Network (CNN)** model to classify **benign and malignant skin lesions** using high-resolution images.
   - Melanoma is a **highly dangerous form of skin cancer**, where early and accurate detection significantly improves patient outcomes.
   - Existing smartphone applications for melanoma detection **often lack accuracy and reliability**, with some misclassifying up to 30% of melanomas as low-risk lesions.
   - This model provides a **reliable, AI-driven diagnostic tool** that outperforms many smartphone-based solutions.

2. **Data Understanding & Preprocessing**
   - The **Melanoma Cancer Image Dataset** contains **13,900 images** labeled as **benign** or **malignant**.
   - Images were **resized to 224x224 pixels** and **normalized** (pixel values between 0 and 1) for consistent training.
   - No missing values or corrupted images were detected.
   - A **balanced dataset** with similar distributions of benign and malignant images ensures unbiased model training.

3. **Model Architecture & Implementation**
   - **CNN Model with 4 convolutional layers** (16, 32, 48, 64 filters), each followed by **MaxPooling layers**.
   - A **fully connected dense layer** (64 neurons, ReLU activation) and an **output layer** (1 neuron, Sigmoid activation).
   - The **Adam optimizer** and **binary cross-entropy loss function** were used for training.
   - **Training set:** Used for learning patterns, **Validation set:** Used for hyperparameter tuning.
   - **Epochs set to 20** based on validation performance to avoid overfitting.

4. **Model Performance & Evaluation**
   - **Test Accuracy:** **91.20%** – indicating **strong classification capability**.
   - **Confusion Matrix Results:**
     - **Benign:** **920 True Negatives**, **80 False Positives**.
     - **Malignant:** **904 True Positives**, **96 False Negatives**.
   - **AUC-ROC Score:** **0.97**, demonstrating **excellent discrimination between benign and malignant lesions**.
   - **Training vs. Validation Loss:** No significant overfitting observed, ensuring model generalization.

5. **Comparisons with Existing Melanoma Detection Technologies**
   - The CNN model outperforms **many smartphone-based melanoma detection applications** that have misclassification rates of up to 30%.
   - The **rigorous training and validation process**, coupled with a high-quality dataset, ensures better reliability than unverified AI-driven smartphone tools.
   - The **availability of a high-accuracy diagnostic model** provides valuable **decision-support for dermatologists** and **self-assessment tools for individuals**.

### 2. Conclusions & Real-World Applications

1. **Significance for Healthcare & Early Diagnosis**
   - This **CNN-based melanoma detection model** provides an **AI-assisted diagnostic tool** that can improve early detection rates and reduce misdiagnosis.
   - **Early detection** of melanoma **dramatically increases survival rates**, making this model an important **clinical decision-support system**.
   - The model's **high accuracy (91.20%) and strong ROC-AUC performance (0.97)** make it a reliable complement to dermatological examinations.

2. **Deployment & Cloud Integration**
   - The trained model is **deployed on Google Cloud Platform (GCP)** for **scalability, efficiency, and accessibility**.
   - GCP provides **robust computational resources** that handle deep learning workloads efficiently.
   - **Advantages over traditional smartphone apps:**
     - **More accurate and reliable** due to **high-quality training data**.
     - **No reliance on variable user-captured images**, improving prediction consistency.
     - **Cloud scalability** allows real-time AI-driven medical assistance.

3. **Potential Use Cases & Future Improvements**
   - **Clinical Use:** A **decision-support tool for dermatologists** to assist in melanoma diagnosis.
   - **Public Health Screening:** Can be integrated into **telemedicine applications** for early detection.
   - **Educational Tool:** Medical students and researchers can use it for **learning AI-assisted diagnostics**.
   - **Further Enhancements:**
     - Fine-tuning the model using **additional datasets** to improve accuracy.
     - Implementing **explainability methods (e.g., Grad-CAM)** to visualize decision-making.
     - Expanding to a **multi-class classification model** for various skin diseases.

4. **Final Thoughts**
   - This project **demonstrates the power of deep learning** in **real-world medical applications**.
   - A structured **CRISP-DM approach** ensures **methodological rigor**, from problem definition to model deployment.
   - By integrating **AI into medical diagnostics**, this work contributes to **early melanoma detection**, potentially saving lives through timely intervention.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request to improve the project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Developed by Jovan Thompson as part of a data science portfolio project.

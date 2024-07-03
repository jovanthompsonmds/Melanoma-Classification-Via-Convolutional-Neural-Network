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

## Insights
The CNN model demonstrated high performance, significantly outperforming existing smartphone apps for melanoma detection. The confusion matrix showed a balanced classification of benign and malignant lesions, and the ROC-AUC score of 0.97 indicates excellent discrimination capability. This project provides a scalable and reliable tool for early melanoma detection, aiding healthcare professionals and patients in timely diagnosis and treatment.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request to improve the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
Developed by Jovan Thompson as part of a data science portfolio project.

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
- Load the data
- Perform linear regression
- Visualize the results
- Analyze the relationship between midterm and final exam grades

## Insights
The analysis shows a moderately strong positive linear relationship between midterm and final grades, with an R-squared value of 0.613. The predictive model can help educators identify students who may need extra support and provide data-driven insights for improving academic strategies.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request to improve the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
Developed by Jovan Thompson as part of a data science portfolio project.

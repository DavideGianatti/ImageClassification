February 2023

## Fruit Image Classification
This repository contains the implementation of various Convolutional Neural Network (CNN) architectures and classifiers (such as Support Vector Machines) for image classification tasks.
The models are trained, evaluated, and saved using TensorFlow, and several classifiers are implemented using scikit-learn.

### Project Structure
The project is organized as follows:
- **extract_features.py**: contains functions to pre-process images and extract low-level features;
- **classifiers.py**: implements methods exploiting different classifiers (linear SVM, polynomial SNM, radial basis function SVM, minimum distance classifier)
- **cnn.py**: implements, trains and saves different architectures of CNNs;
- **last_layer.py**: extracts features from the last hidden layer of a CNN and trains classifiers on these features;
- **low_level_feature.py**: trains a classifier from classifiers.py using low-level features.

### Data Set
The training, validation and test sets can be downloaded from https://drive.google.com/file/d/1WuRFy2xr7IYie-SdHYgj3XrgfWUH7W1Q/view. 
Collect them in a directory named **fruits**.

### Report
The interest reader can delve into the project methodologies and results by reviewing the PDF report: **fruits_classification_report.pdf**.


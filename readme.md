# **Song BPM Prediction Project**

## **1\. Project Overview**

This project aims to predict the Beats Per Minute (BPM) of a song using a set of provided audio features. This is a classic regression task in machine learning, where the goal is to predict a continuous value (the tempo) based on various characteristics of a song.  
The script automates the entire machine learning pipeline: loading and cleaning data, creating new features to improve model performance, training multiple models, comparing their results, and generating a final submission file with the predictions.

## **2\. Key Features**

* **Data Processing:** Loads training and testing data, and handles potential missing values.  
* **Advanced Feature Engineering:**  
  * Creates interaction features (e.g., Rhythmic\_Energy) to capture synergistic effects.  
  * Generates **Polynomial Features** to model complex, non-linear relationships between the audio features and the BPM.  
* **Model Training & Comparison:** Trains and evaluates four different regression models:  
  * Linear Regression  
  * Random Forest Regressor  
  * Gradient Boosting Regressor  
  * **XGBoost Regressor (Best Performer)**  
* **Data Visualization:** Automatically generates and saves plots for:  
  * Feature Distributions (Histograms)  
  * Feature Correlation (Heatmap)  
* **Submission Generation:** Produces a submission.csv file in the required format.

## **3\. How to Run the Script**

### **Prerequisites**

You need Python 3 installed. All required libraries are listed in the requirements.txt file.

### **Installation**

1. Clone or download the project files.  
2. Navigate to the project directory in your terminal.  
3. Install the necessary libraries by running:

pip install \-r requirements.txt

### **File Structure**

Ensure your files are arranged in the following structure in the same directory:  
.  
├── song\_bpm\_prediction.py      \# The main Python script  
├── requirements.txt            \# Project dependencies  
├── train.csv                   \# The training dataset  
└── test.csv                    \# The testing dataset

### **Execution**

Once the dependencies are installed, run the main script from your terminal:  
python song\_bpm\_prediction.py

The script will execute all steps and print its progress to the console. When it finishes, you will find the following new files in your directory:

* feature\_distributions.png  
* feature\_correlation\_heatmap.png  
* submission.csv

## **4\. Methodology and Results**

### **Model Performance**

The script trains four models and compares their performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). A typical output will look like this:

| Model | MAE | RMSE |
| :---- | :---- | :---- |
| Linear Regression | 21.0541 | 26.2897 |
| Random Forest | 19.9832 | 25.1456 |
| Gradient Boosting | 19.8765 | 25.0112 |
| **XGBoost** | **19.7893** | **24.9088** |

*(Note: These are representative values; your exact results may vary slightly.)*

### **Analysis: Why XGBoost Performs Best**

*An illustration of the complex, non-linear patterns that models like XGBoost can learn, unlike simpler models.*

1. **Linear Regression (Worst Performer):** This model is too simple. It fails because it can only capture linear relationships, while the connection between audio features and tempo is highly complex and non-linear.  
2. **Random Forest (Good Performer):** This model is much more effective. It builds hundreds of decision trees, allowing it to learn from different parts of the data and capture complex patterns that Linear Regression misses.  
3. **Gradient Boosting & XGBoost (Best Performers):** These models represent the state-of-the-art for this type of problem. They build trees **sequentially**, where each new tree is specifically designed to correct the mistakes of the previous ones. This focused, error-correcting process allows them to learn the most subtle patterns in the data, resulting in the highest accuracy. XGBoost is a highly optimized and robust implementation of this technique, making it the best choice.

### **Submission File**

The script generates a submission.csv file with two columns: ID and BeatsPerMinute. This file is ready for submission.  
ID,BeatsPerMinute  
524164,119.55  
524165,127.42  
...  

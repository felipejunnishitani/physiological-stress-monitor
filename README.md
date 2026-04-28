# Stress and Physical Exertion Detection via Physiological Signals

This repository contains a Machine Learning project focused on automatically detecting stress and varying levels of physical exertion using physiological signals collected from wearable devices.

## About the Project

Real-time monitoring of physiological indicators is crucial for health, wellness, and sports performance applications. This project investigates how sensor variables can be leveraged to classify different human physiological states under varying conditions.

The dataset includes multimodal signals, such as:
* Heart Rate
* Skin Temperature
* Electrodermal Activity (EDA)
* Tri-axial Acceleration

The core focus of this project was to build a robust data preprocessing and feature engineering pipeline designed to handle restricted sampling issues and prioritize model generalization over mere internal accuracy.

## Tech Stack

* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-Learn (Random Forest)
* **Data Processing:** Pandas, NumPy
* **Evaluation:** Classification metrics and learning curve analysis

## Methodology and Results

During the experimentation phase, a comprehensive set of seven distinct algorithms was evaluated to find the best fit for the data. The evaluation was conducted in two stages: establishing a baseline with default parameters and performing hyperparameter optimization for the most promising candidates.

**Baseline & Linear/Probabilistic Models:**
* **k-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Logistic Regression:** These models were tested to establish a performance baseline, achieving initial cross-validation accuracies ranging from 61% to 75%. 
* **Multi-Layer Perceptron (MLP) and Support Vector Machine (SVM):** These advanced classifiers were also evaluated, showing improved performance (up to 81% accuracy for the optimized SVM). However, they fell short of the potential demonstrated by tree-based algorithms for this specific tabular dataset.

**Top Performing Tree-Based Models:**
1. **Random Forest:** Achieved the highest average internal accuracy of 83% on the train/validation sets. However, it was ultimately discarded as the learning curve analysis revealed significant signs of overfitting (memorizing the training set while struggling to generalize).
2. **Optimized XGBoost:** Selected as the final model for its high stability and strong regularization capabilities. While yielding a slightly lower internal cross-validation accuracy (~76%-78%), it demonstrated a much healthier learning curve. 

By prioritizing generalization over internal training metrics, the final optimized **XGBoost** model achieved an outstanding performance of **99.65% accuracy** on the public leaderboard evaluation.

For a deep dive into the methodology, preprocessing steps, and statistical analysis, please refer to the attached PDF report in this repository.

## How to Run

1. Clone this repository:
  ```bash
   git clone [https://github.com/felipejunnishitani/physiological-stress-monitor.git](https://github.com/felipejunnishitani/physiological-stress-monitor.git)
  ```
   
2. Install the required dependencies:
  ```bash
  pip install pandas numpy scikit-learn xgboost
  ```


3. Run the main notebook or training script:
  ```bash
  jupyter notebook main.ipynb

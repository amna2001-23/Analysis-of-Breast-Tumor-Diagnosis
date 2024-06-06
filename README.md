# Predictive Analysis of Breast Tumor Diagnosis

## Introduction
This project aims to predict whether a breast cancer tumor is malignant or benign using machine learning techniques. The dataset used is from breast fine-needle aspiration (FNA) tests.

## Problem Statement
Breast cancer is the most common malignancy among women, and it is the second leading cause of cancer death among women. This project builds a model to classify a breast cancer tumor using two classifications:
- 1 = Malignant (Cancerous) - Present
- 0 = Benign (Not Cancerous) - Absent

## Expected Outcome
The model should accurately classify the tumors as malignant or benign based on the given data.

## Objectives of Data Exploration
Exploratory Data Analysis (EDA) is performed to understand the nature of the data, its distribution, and interrelationships within the dataset. The EDA techniques used include:
- Histograms
- Density Plots
- Box and Whisker Plots

## Pre-Processing the Data
Data preprocessing involves:
- Assigning numerical values to categorical data
- Handling missing values
- Normalizing the features

## Predictive Model
The Support Vector Machine (SVM) algorithm is used to build the predictive model.

## Model Accuracy
Model accuracy is evaluated using the Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC).

## Automate the ML Process using Pipelines
Pipelines are used to automate standard workflows in a machine learning project.

## Summary
The steps covered include:
- Problem Definition
- Loading the Dataset
- Analyzing Data
- Evaluating Algorithms
- Algorithm Tuning
- Finalizing the Model

## Installation
To run this project, install the necessary libraries listed in `requirements.txt`.

## Usage
Run the Streamlit app to view the interactive dashboard:
```sh
streamlit run breast.py

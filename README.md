# Spam Detection and Data Loss Prevention (DLP) System

This project involves building a spam detection and data loss prevention (DLP) system usinf machine learning. The goal is to predict whether an email is spam and trigger an alert for potential data leakage.


## Overview

The project utilizes several machine learning models to classify emails as spam or not. It includes a process for:

- Text preprocessing using TF-IDF Vectorization
- Training models (Logistic Regression, SVM, and Random Forest) on email data
- Evaluating model performance
- Deploying a DLP system that detects potential data leakage in emails

The models are trained on a spam dataset, and predictions are made based on incoming email text. If the system detects potential data leakage (spam), an alert is triggered.

## Features

- **Text Preprocessing**: Using TF-IDF Vectorization for transforming email content into numerical format.
- **Model Training**: Trains three machine learning models: Logistic Regression, Support Vector Machine (SVM), and Random Forest.
- **Model Prediction**: Predicts whether an email is spam using the trained models.
- **DLP Alerts**: Trigger alerts for potential data leakage when spam is detected.

## Technologies Used

- **Python**: The core language for data processing and model building.
- **Scikit-learn**: For machine learning algorithms (Logistic Regression, SVM, Random Forest).
- **Pandas**: For data manipulation and loading datasets.
- **TF-IDF Vectorization**: For text preprocessing and feature extraction.
- **Pickle**: To save trained models and vectorizer.
- **Smtplib**: To integrate email functionality for real-time predictions and alerts.

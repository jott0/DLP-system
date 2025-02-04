import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import smtplib

"""# **LOAD DATA**"""

from google.colab import files
uploaded = files.upload()

# Load the dataset
df = pd.read_csv('spam.csv',encoding='latin-1')

# Split the data into training and testing sets
X = df['v2']
y = df['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# **PREPROCESS DATA**"""

# Preprocess text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Save the vectorizer
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

"""# **TRAIN MODELS**"""

# Train Logistic Regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train_vectorized, y_train)
y_pred = logistic_regression_model.predict(X_test_vectorized)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

# Train SVM model
svm_model = SVC()
svm_model.fit(X_train_vectorized, y_train)
y_pred = svm_model.predict(X_test_vectorized)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))

# Train Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train_vectorized, y_train)
y_pred = random_forest_model.predict(X_test_vectorized)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# Save the models
pickle.dump(logistic_regression_model, open('logistic_regression_model.pkl', 'wb'))
pickle.dump(svm_model, open('svm_model.pkl', 'wb'))
pickle.dump(random_forest_model, open('random_forest_model.pkl', 'wb'))

"""# **DLP AND PREDICTIONS**"""

# Email ingestion and preprocessing
def ingest_email(email_text):
    email_vectorized = vectorizer.transform([email_text])
    return email_vectorized

# ML model prediction
def predict_email(email_vectorized):
    predictions = []
    predictions.append(logistic_regression_model.predict(email_vectorized)[0])
    predictions.append(svm_model.predict(email_vectorized)[0])
    predictions.append(random_forest_model.predict(email_vectorized)[0])
    return predictions

# Alert system
def trigger_alert(email_text, predictions):
    print("Predictions:")
    print("Logistic Regression:", predictions[0])
    print("SVM:", predictions[1])
    print("Random Forest:", predictions[2])
    if any(prediction == 'spam' for prediction in predictions):
        print("Alert: Potential data leakage detected in email:", email_text)
        print("Data Leakage Alert!")

# Test the DLP system
email_text = "Sorry my roommates took forever, it ok if I come by now?"
email_vectorized = ingest_email(email_text)
predictions = predict_email(email_vectorized)
trigger_alert(email_text, predictions)

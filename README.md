ChurnGuard AI - Telecom Customer Churn Predictor
A machine learning web application built for the Horizon competition. This project predicts whether a telecom customer is at risk of churning (leaving the company) based on their billing patterns, tenure, and demographics.

What this does
This is an end-to-end data science pipeline wrapped in a clean web interface. It has two main features:

Individual Analysis: Input a single customer's details to get a real-time risk probability score.

Batch Processing: Upload a CSV file containing hundreds of customers. The app will process them all, scale the data, run the model, and generate a downloadable prediction report.

Tech Stack
Machine Learning: Scikit-learn (Support Vector Classifier), Pandas, Numpy, Joblib

Web Framework: Streamlit

Core Language: Python

How to run it locally
Make sure you have the required Python libraries installed:
pip install streamlit joblib scikit-learn pandas numpy

Ensure the model files (best_churn_model.pkl and scaler.pkl) are in the same directory as your app.py script.

Boot up the local server:
streamlit run app.py

Developer Note & Transparency
I am primarily a full-stack web developer used to working with the MERN stack (MongoDB, Express, React, Node). Diving into deep Python data science and machine learning was a completely new style of coding for me, and I built this specifically to push my limits for Horizon.

To be 100% transparent about the development process:

The Core Math & Backend (Done by me): I am the one behind the core machine learning code. I handled the data cleaning, the Jupyter notebooks, the feature scaling, and the model training. I specifically tuned the Support Vector Machine to handle imbalanced data, optimized the Precision-Recall tradeoff, and wrote the prediction logic routing inside the app.py file.

The Frontend UI (AI Assisted): Because Streamlit is a completely unfamiliar library to me (I normally use React for interfaces), I used AI to help me structure the frontend code. I used it to understand how to format the tabs, columns, and metric boxes so that my backend Python model actually had a decent looking UI to sit inside.

I focused my energy on learning the actual machine learning math from scratch, and used AI to help me wrap it in a UI so I could ship a finished product.
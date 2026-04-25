# ChurnGuard Ai
## I built this project to learn how to train a machine learning model in a Jupyter Notebook and actually deploy it to a real web interface so people can use it.

### What ChurnGuard AI actually does
It is a simple dashboard that helps businesses figure out which customers are about to cancel their service (churn). 

* Single Check: You can type in a single customer's details (like age, how long they've been subscribed, and their monthly bill) to get an instant risk score almost instantly thanks to streamlit.
* Batch Check: If you have a whole spreadsheet of customers, you can upload a CSV file. The app will analyze all of them at once and generate a downloadable report showing exactly who is at "High Risk" of leaving so the business can try to keep them.

### How I Built It
1. I first of all found this dataset in kaggle downloaded it and then explored the things i could built with this.
2.I trained a Support Vector Machine model and exported it using Joblib `best_churn_model.pkl` and `scaler.pkl`).
3.Then after training the data i made the `app.py` for the deployment of this so others can use it as well.


### How to run this LocallyIf you want to run this project on your own machine:

1. Clone this repository.
2. Make sure you have the required libraries installed(here below):
`pip install streamlit pandas scikit-learn numpy joblib`
3. Run the Streamlit app:
`streamlit run app.py`


### HERE IS THE VIDEO TO SEE IT IN ACTION


https://github.com/user-attachments/assets/2236d571-a6e4-4607-80ce-9fea5e499857



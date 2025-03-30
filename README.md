# Email Spam Classification using Machine Learning

## **Overview**

This project aims to classify emails as **spam** or **non-spam (ham)** using various machine learning techniques. The dataset consists of labeled emails, and the goal is to build a predictive model that accurately identifies spam emails.

## **Features**

- Data preprocessing (cleaning and transformation)
- Feature extraction (TF-IDF, Bag of Words, etc.)
- Model training using multiple algorithms (Naïve Bayes, SVM, Random Forest, etc.)
- Evaluation of model performance
- Deployment of the trained model

## **Dataset**

We use the **SpamAssassin Public Corpus** or the **UCI Machine Learning Repository's SMS Spam Collection** for training and testing. The dataset contains:

- **Label**: spam or ham
- **Email text**

## **Technologies Used**

- **Python**
- **Scikit-learn**
- **Pandas & NumPy**
- **NLTK (Natural Language Toolkit)**
- **Flask** (for deployment)

## **Installation**

1. Clone the repository: https://github.com/princekumar72550/Email-Spam-Classifier
2. Navigate to the project directory: cd email-spam-classification
3. Install dependencies: pip install -r requirements.txt

## **Usage**

1. Preprocess the dataset: python preprocess.py
2. Train the model: python train.py
3. Test the model: python test.py
4. 4. Run the web application: python app.py


## **Model Performance**

The model is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

## **Flowchart**

Below is the flowchart illustrating the email spam classification process:

              +------------------+
              |  Load Dataset    |
              +------------------+
                      |
                      v
              +------------------+
              | Data Preprocessing |
              +------------------+
                      |
                      v
              +------------------+
              | Feature Extraction |
              +------------------+
                      |
                      v
              +------------------+
              | Model Training   |
              +------------------+
                      |
                      v
              +------------------+
              | Model Evaluation |
              +------------------+
                      |
                      v
              +------------------+
              |  Prediction      |
              +------------------+
                      |
                      v
              +------------------+
              |  Deployment      |
              +------------------+

## **Future Enhancements**

- Implement deep learning models (LSTMs, Transformers)
- Improve feature extraction techniques
- Deploy the model as a web service

## **Contributing**

Contributions are welcome! Feel free to fork the repository and submit a pull request.

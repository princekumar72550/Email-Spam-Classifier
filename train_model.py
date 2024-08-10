import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example data (replace with your actual data)
emails = [
    "Free entry in 2 a weekly competition to win FA Cup final tickets, text FA to 87121 to receive entry question(std txt rate)",
    "Nah I don't think he goes to usf, he lives around here though",
    "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
    "Had your mobile 11 months or more? You are entitled to update to the latest colour mobiles with camera for free! Call the mobile update co free on 08002986030",
    "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."
]
labels = [1, 0, 1, 1, 0]  # 1 for spam, 0 for not spam

# Create the vectorizer and transform the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# Save the vectorizer and the model
with open('vectorizer.plk', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('Naive_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer saved successfully.")

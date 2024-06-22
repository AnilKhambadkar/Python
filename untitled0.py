corpus = [
    ("no issues with screenshot.", "yes"),
    ("Sure, you can take a screenshot.", "yes"),
    ("Yes, feel free to capture it.", "yes"),
    ("Absolutely, go ahead.", "yes"),
    ("No problem, take the screenshot.", "yes"),
    ("I don't mind, you can take it.", "yes"),
    ("Yes, you have my permission.", "yes"),
    ("Go ahead, take a screenshot.", "yes"),
    ("Of course, take the screenshot.", "yes"),
    ("Fine by me, capture it.", "yes"),
    ("No issues, take a screenshot.", "yes"),
    ("Please, go ahead and take it.", "yes"),
    ("Yes, take it now.", "yes"),
    ("Yes, you can.", "yes"),
    ("Yes, go for it.", "yes"),
    ("Sure thing, take it.", "yes"),
    ("Feel free to take a screenshot.", "yes"),
    ("No, I don't want you to take a screenshot.", "no"),
    ("I'd prefer if you didn't take a screenshot.", "no"),
    ("Please don't take a screenshot.", "no"),
    ("I do not give permission to take a screenshot.", "no"),
    ("I would rather you not take a screenshot.", "no"),
    ("Can you not take a screenshot, please?", "no"),
    ("I'd prefer if you didn't.", "no"),
    ("Don't take a screenshot.", "no"),
    ("No screenshots, please.", "no"),
    ("I don't want any screenshots taken.", "no"),
    ("I'd rather not have a screenshot taken.", "no"),
    ("Please avoid taking a screenshot.", "no"),
    ("No, please respect my privacy.", "no"),
    ("Sorry, but no screenshots.", "no"),
    ("I'd rather you not.", "no"),
    ("I don't feel comfortable with that.", "no"),
    ("No, that's not okay.", "no"),
    ("No, I would rather not.", "no"),
    ("Not this time, please.", "no"),
    ("No, I’d rather keep it private.", "no")
]

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Convert the corpus into a DataFrame
df = pd.DataFrame(corpus, columns=["statement", "label"])

# Prepare the data
X = df["statement"]
y = df["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and MultinomialNB
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Function to classify new input
def classify_permission(input_text):
    prediction = model.predict([input_text])
    return prediction[0]

# Example usage
user_input = input("Please enter your response: ")
classification = classify_permission(user_input)

print(f"Classification: {classification}")

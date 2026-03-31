# Simple Sentiment Analyzer (Positive / Negative)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Small training dataset
# Some example sentences labelled as positive (1) or negative (0)

train_sentences = [
    "I love this product",
    "This is a great movie",
    "I am very happy today",
    "What a fantastic experience",
    "The food was amazing",
    "I enjoyed this a lot",
    "This is the best day",

    "I hate this",
    "This is a terrible movie",
    "I am very sad today",
    "What a bad experience",
    "The food was horrible",
    "I did not enjoy this",
    "This is the worst day"
]

train_labels = [
    1, 1, 1, 1, 1, 1, 1,   # positive = 1
    0, 0, 0, 0, 0, 0, 0    # negative = 0
]

# 2. Convert text to numbers
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_sentences)

# 3. Train a simple classifier
model = MultinomialNB()
model.fit(X_train, train_labels)

print("Simple Sentiment Analyzer")
print("Type a sentence and I will tell if it is Positive or Negative.")
print("Type 'exit' to quit.\n")

# 4. Take user input and predict
while True:
    user_input = input("Enter a sentence: ")

    if user_input.lower() in ["exit", "quit", "q"]:
        print("Exiting... Bye!")
        break

    # Transform the input using the same vectorizer
    X_test = vectorizer.transform([user_input])
    prediction = model.predict(X_test)[0]
    prob = model.predict_proba(X_test)[0]

    if prediction == 1:
        label = "Positive 😊"
        confidence = prob[1]
    else:
        label = "Negative 😞"
        confidence = prob[0]

    print(f"Prediction: {label} (confidence: {confidence:.2f})\n")

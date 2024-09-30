import tarfile
import urllib.request
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

# Step 1: Download and extract data
url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
data_path = './rt-polaritydata'

if not os.path.exists(data_path):
    urllib.request.urlretrieve(url, 'rt-polaritydata.tar.gz')
    with tarfile.open('rt-polaritydata.tar.gz', 'r:gz') as tar:
        tar.extractall()

# Step 2: Load the data
def load_data():
    with open(f'{data_path}/rt-polarity.pos', 'r', encoding='latin-1') as pos_file:
        positive_sentences = pos_file.readlines()
    with open(f'{data_path}/rt-polarity.neg', 'r', encoding='latin-1') as neg_file:
        negative_sentences = neg_file.readlines()
    
    return positive_sentences, negative_sentences

positive_sentences, negative_sentences = load_data()

# Step 3: Create train, validation, test sets
train_pos, val_pos, test_pos = positive_sentences[:4000], positive_sentences[4000:4500], positive_sentences[4500:]
train_neg, val_neg, test_neg = negative_sentences[:4000], negative_sentences[4000:4500], negative_sentences[4500:]

train_data = train_pos + train_neg
val_data = val_pos + val_neg
test_data = test_pos + test_neg

train_labels = [1]*4000 + [0]*4000
val_labels = [1]*500 + [0]*500
test_labels = [1]*831 + [0]*831

# Step 4: Preprocessing (tokenization, vectorization)
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_data)
X_val = vectorizer.transform(val_data)
X_test = vectorizer.transform(test_data)

# Step 5: Function to evaluate models
def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_val, val_predictions).ravel()
    
    # Precision, Recall, F1-score
    precision = precision_score(y_val, val_predictions)
    recall = recall_score(y_val, val_predictions)
    f1 = f1_score(y_val, val_predictions)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("="*40)
    
    return precision, recall, f1

# Step 6: Try multiple models
models = [
    LogisticRegression(),
    SVC(),
    RandomForestClassifier(),
    MultinomialNB()
]

best_model = None
best_f1 = 0

for model in models:
    precision, recall, f1 = evaluate_model(model, X_train, train_labels, X_val, val_labels)
    
    # Check if current model has the best F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# Step 7: Test on the best model
print(f"\nBest Performing Model: {best_model.__class__.__name__}")

best_model.fit(X_train, train_labels)
test_predictions = best_model.predict(X_test)

# Test results
tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel()
precision = precision_score(test_labels, test_predictions)
recall = recall_score(test_labels, test_predictions)
f1 = f1_score(test_labels, test_predictions)

print(f"Test Results for {best_model.__class__.__name__}:")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

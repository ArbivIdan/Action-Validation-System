import os.path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report
import numpy as np
from torch.nn import functional as F


class TextClassifier:
    def __init__(self, model_checkpoint="results-distilbert/checkpoint-468", tokenizer_name="distilbert-base-uncased", max_length=128):
        # Load the trained model and tokenizer
        model_checkpoint_path = os.path.join(os.getcwd(), f'classifiers/{model_checkpoint}')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def prepare_input(self, texts):
        """Tokenizes the input texts."""
        encoding = self.tokenizer(
            texts,
            padding=True,  # Padding to ensure all inputs have the same length
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"  # This ensures the output is in PyTorch tensor format
        )
        return encoding

    def classify(self, texts):
        """Performs inference on the given texts."""
        # Prepare the input
        inputs = self.prepare_input(texts)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the predicted classes (0 or 1)
        predicted_classes = torch.argmax(outputs.logits, dim=-1).tolist()
        return np.array(predicted_classes)

    def generate_report(self, df_test, predictions):
        """Generates a classification report."""
        # Convert the true labels to a numpy array
        y_true = df_test["Label"].values
        # Generate the classification report
        return classification_report(y_true, predictions, target_names=["0 (invalid)", "1 (valid)"])

    def evaluate(self, df):
        """Evaluates the model on the given dataframe."""
        # Split into train and test sets
        df_train, df_test = train_test_split(
            df,
            test_size=0.2,  # 20% test set
            stratify=df["Label"],  # Maintain label distribution
            random_state=42
        )

        # Tokenize and classify the test set
        texts = df_test["Text"].tolist()
        predictions = self.classify(texts)

        # Generate and print the classification report
        report = self.generate_report(df_test, predictions)
        print("\nClassification Report on Test (Original Distribution):")
        print(report)

    def predict_with_probabilities(self, text):
        """Predicts the label and probability for a given text."""
        # Prepare the input
        inputs = self.prepare_input([text])

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply softmax to the logits to get probabilities
        probs = F.softmax(outputs.logits, dim=-1)

        # Get the predicted class (0 or 1) and the probability for that class
        predicted_class = torch.argmax(probs, dim=-1).item()
        probability = probs[0, predicted_class].item()

        return predicted_class, probability

# Example Usage:

# Initialize the classifier
classifier = TextClassifier()
# Load the dataset
# df = pd.read_csv("../data/action_enrichment_ds_home_exercise.csv")  # columns: [EventName, Text, Label, action]
# df = df.dropna(subset=["Text"])  # Ensure no missing values in these columns

print(classifier.predict_with_probabilities("You may see some double teams from the wars that you gotta rotate out of it."))

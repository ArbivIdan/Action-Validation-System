from typing import Tuple

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_curve, auc, confusion_matrix, accuracy_score, classification_report
)

from classifiers.sentence_to_vec import vectorize_tfidf_batch


class BinaryClassifierEvaluator:
    def __init__(self):
        """
        Initialize evaluator with ground truth labels, predicted labels, and optional probabilities.
        """
        self.best_model = joblib.load('best_model.pkl')
        # self.y_true = np.array(y_true)
        # self.y_pred = np.array(y_pred)
        # self.y_prob = np.array(y_prob) if y_prob is not None else None

    def recall(self):
        """Calculate recall (sensitivity)."""
        return recall_score(self.y_true, self.y_pred)

    def precision(self):
        """Calculate precision."""
        return precision_score(self.y_true, self.y_pred)

    def f1_score(self):
        """Calculate F1-score."""
        return f1_score(self.y_true, self.y_pred)

    def accuracy(self):
        """Calculate accuracy."""
        return accuracy_score(self.y_true, self.y_pred)

    def confusion_matrix(self):
        """Return confusion matrix as a NumPy array."""
        return confusion_matrix(self.y_true, self.y_pred)

    def roc_auc(self, plot=True):
        """Compute ROC curve and AUC score, optionally plot the curve."""
        if self.y_prob is None:
            raise ValueError("y_prob is required for ROC and AUC calculations.")

        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)

        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.show()

        return roc_auc

    def evaluate_all(self):
        """Print all metrics at once."""
        print(f"Accuracy: {self.accuracy():.2f}")
        print(f"Precision: {self.precision():.2f}")
        print(f"Recall: {self.recall():.2f}")
        print(f"F1 Score: {self.f1_score():.2f}")
        print(f"AUC: {self.roc_auc(plot=False):.2f}")
        print("Confusion Matrix:")
        print(self.confusion_matrix())

    def get_validity(self, sentence: str) -> Tuple[int, float]:
        vectorized_sentence = vectorize_tfidf_batch([sentence])
        y_pred = self.best_model.predict(vectorized_sentence)
        sentence_prob = self.best_model.predict_proba(vectorized_sentence)
        return y_pred, sentence_prob

if __name__ ==  "__main__":
    x_test = "It's just a great backdoor cut."
    binary_classifier = BinaryClassifierEvaluator()
    print(binary_classifier.get_validity(x_test))

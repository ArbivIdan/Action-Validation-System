import os.path
import pickle

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
import seaborn as sns
from classifiers.torch_models import LSTMModel, MLP


def plot_tsne(X, y):
    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    X_embedded = tsne.fit_transform(X)
    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette=["red", "blue"], alpha=0.7)
    plt.title("t-SNE Visualization of Text Data")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Label", labels=["Class 0", "Class 1"])
    plt.show()

# Example function to train and evaluate an MLP classifier
def train_mlp(X_train, X_test, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("MLP Classifier Results:")
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(report)
    return report['macro avg']


# Example function to train and evaluate a Gradient Boosting classifier
def train_gradient_boosting(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Gradient Boosting Results:")
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(report)
    return report['macro avg']


def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Random Forest Results:")
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(report)
    return report['macro avg']


def train_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear')  # You can try 'rbf' kernel as well
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("SVM Results:")
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(report)
    return report['macro avg']


def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Logistic Regression Results:")
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    joblib.dump(model, "best_model.pkl")
    print(report)
    return report['macro avg']


def train_torch_classifier(X_train, X_test, y_train, y_test, model_type, input_dim):
    # Reshape the data based on the model type
    if model_type == 'LSTM':
        # LSTM requires input of shape (batch_size, seq_len, input_dim)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, input_dim)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, input_dim)
    elif model_type == 'FFNN':
        # FFNN can directly use the input shape (batch_size, input_dim)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Convert target labels to tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensure it's the correct shape
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Initialize the model
    if model_type == 'LSTM':
        model = LSTMModel(input_dim=input_dim)
    elif model_type == 'FFNN':
        model = MLP(input_dim=input_dim)

    # Define loss and optimizer
    criterion = torch.nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print loss for every epoch
        if (epoch + 1) % 2 == 0:  # Print every 2 epochs
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze().cpu().numpy()
        y_pred = (y_pred >= 0.5).astype(int)  # Convert probabilities to binary labels

    print(f"Classification Report for {model_type}:")
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(report)
    return report['macro avg']


def upsample_input(X_train, y_train):
    # Separate the classes for upsampling
    X_train_0 = X_train[y_train == 0]
    y_train_0 = y_train[y_train == 0]
    X_train_1 = X_train[y_train == 1]
    y_train_1 = y_train[y_train == 1]

    # Upsample the minority class (0's) to match the size of the majority class (1's)
    X_train_0_upsampled, y_train_0_upsampled = resample(X_train_0, y_train_0,
                                                        replace=True,  # Sample with replacement
                                                        n_samples=X_train_1.shape[0],  # Match the majority class size
                                                        random_state=42)

    # Combine the upsampled 0's with the 1's
    X_train_balanced = np.vstack((X_train_1, X_train_0_upsampled))  # No need to call toarray()
    y_train_balanced = np.concatenate((y_train_1, y_train_0_upsampled))

    print(f"Number of 0 examples after resampling: {np.sum(y_train_balanced == 0)}")
    print(f"Number of 1 examples after resampling: {np.sum(y_train_balanced == 1)}")

    return X_train_balanced, y_train_balanced

# Load data
action_enrichment_df = pd.read_csv(r"../data/action_enrichment_ds_home_exercise.csv")
sentences = action_enrichment_df["Text"].astype(str).drop_duplicates()
y = action_enrichment_df.loc[sentences.index, "Label"]
best_model_info = {'vector_type': None, 'vec_name': None, 'model': None, 'f1-score': 0}
vector_types = ["default", "preprocessed", "augment"]
vector_types = ["default"]
for vector_type in vector_types:
    # Load vectors_dict from pickle file
    with open(os.path.join(os.getcwd(),f'vectors_data/vectors_dict_{vector_type}.pkl'), 'rb') as f:
        vectors_dict = pickle.load(f)

    # Loop through all vectors and train models
    for vec_name, X in vectors_dict.items():
        print(f"\nTraining with {vec_name}...")

        # Ensure X is 2D
        X = np.array(X)
        y = np.array(y)
        # plot_tsne(X, y)
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data before polynomial transformation
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        # Optionally: Upsample the training data
        X_train, y_train = upsample_input(X_train, y_train)
        input_dim = X_train.shape[1]  # Number of features

        models = {
            # 'MLP': train_mlp(X_train, X_test, y_train, y_test),
            # 'Gradient Boosting': train_gradient_boosting(X_train, X_test, y_train, y_test),
            # 'Random Forest': train_random_forest(X_train, X_test, y_train, y_test),
            # 'SVM': train_svm(X_train, X_test, y_train, y_test),
            'Logistic Regression': train_logistic_regression(X_train, X_test, y_train, y_test),
            # 'FFNN': train_torch_classifier(X_train, X_test, y_train, y_test, model_type="FFNN", input_dim=input_dim),
            # 'LSTM': train_torch_classifier(X_train, X_test, y_train, y_test, model_type="LSTM", input_dim=input_dim),
        }
        # Train and evaluate all models
        for model_name, macro_avg in models.items():
            if macro_avg['f1-score'] > best_model_info['f1-score']:
                best_model_info['vector_type'] = vector_type
                best_model_info['vec_name'] = vec_name
                best_model_info['model'] = model_name
                best_model_info['f1-score'] = macro_avg['f1-score']


print(f"The best model is '{best_model_info['model']}' using vector type '{best_model_info['vector_type']}' with vector name '{best_model_info['vec_name']}' and an F1-score of {best_model_info['f1-score']:.4f}")





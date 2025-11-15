import pandas as pd
import numpy as np
import time

from algorithms import PCA_Model, KNN_Model, LogisticRegressionModel, StackedModel

#metrix
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def recall_macro(y_true, y_pred):
    recalls = []
    classes = np.unique(y_true)
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recalls.append(tp / (tp + fn + 1e-9))
    return np.mean(recalls)

def f1_macro(y_true, y_pred):
    f1s = []
    classes = np.unique(y_true)
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1s.append(f1)
    return np.mean(f1s)

#load data
def load_mnist_csv(path):
    df = pd.read_csv(path)
    y = df.iloc[:, 0].astype(int).values
    X = df.iloc[:, 1:-1].values
    return X, y

#main function

def main():
    print("Loading data...")

    X_train, y_train = load_mnist_csv("MNIST_train.csv")
    X_val, y_val = load_mnist_csv("MNIST_validation.csv")

    print("Shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)

    # Normalize 0–255 to 0–1
    X_train = X_train / 255.0
    X_val = X_val / 255.0

 #pca
    pca = PCA_Model(n_components=100)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)

  #knn
    knn = KNN_Model(k=5)
    logi = LogisticRegressionModel(lr=0.2, epochs=400)
    model = StackedModel(knn, logi)

    print("Training stacked model...")
    start = time.time()
    model.fit(X_train_pca, y_train)
    end = time.time()
    print(f"Training time: {end - start:.2f} sec")


    y_pred_train = model.predict(X_train_pca)
    acc_train = accuracy(y_train, y_pred_train)
    rec_train = recall_macro(y_train, y_pred_train)
    f1_train = f1_macro(y_train, y_pred_train)
    prob_train = model.predict_proba(X_train_pca)

    print("\n===== TRAIN METRICS =====")
    print("Accuracy:", acc_train)
    print("Recall:", rec_train)
    print("F1 Score:", f1_train)

    y_pred_val = model.predict(X_val_pca)
    acc_val = accuracy(y_val, y_pred_val)
    rec_val = recall_macro(y_val, y_pred_val)
    f1_val = f1_macro(y_val, y_pred_val)
    prob_val = model.predict_proba(X_val_pca)

    print("\n===== VALIDATION METRICS =====")
    print("Accuracy:", acc_val)
    print("Recall:", rec_val)
    print("F1 Score:", f1_val)


if __name__ == "__main__":
    main()

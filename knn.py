import numpy as np
import pandas as pd

# Define StandardScaler
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Define LabelEncoder
class LabelEncoder:
    def fit(self, y):
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)

# Define knn (Manhattan Distance and weighted by distance)
class KNNClassifier:
    def __init__(self, n_neighbors=3, distance_metric='manhattan'):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Vectorize distance 
        distances = self._compute_distances(X_test)
        predictions = self._predict_from_distances(distances)
        return predictions

    def _compute_distances(self, X_test):
        if self.distance_metric == 'manhattan':
            # Compute Manhattan Distance with numpy vectorization
            distances = np.abs(X_test[:, np.newaxis] - self.X_train).sum(axis=2)
        return distances

    def _predict_from_distances(self, distances):
        # Find the nearest k neighbors for each sample
        neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neighbors_distances = np.take_along_axis(distances, neighbors_indices, axis=1)
        neighbors_labels = np.take(self.y_train, neighbors_indices)
        
        # Calculate the prob weighted by distance
        weights = 1 / (neighbors_distances + 1e-10)  # Prevent division by 0
        prob_class_1 = np.sum(weights * neighbors_labels, axis=1) / np.sum(weights, axis=1)
        return prob_class_1

# Preprocess data
def preprocess_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Features engineering by label encoding
    label_encoders = {}
    for col in ['Geography', 'Gender']:  # Categorical features
        label_encoders[col] = LabelEncoder()
        train_data[col] = label_encoders[col].fit_transform(train_data[col])
        test_data[col] = label_encoders[col].transform(test_data[col])

    # features and labels
    X_train = train_data.drop(['Exited', 'CustomerId', 'Surname', 'id'], axis=1)
    y_train = train_data['Exited']
    X_test = test_data.drop(['CustomerId', 'Surname', 'id'], axis=1)

    numeric_cols = [col for col in X_train.columns if col not in ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']]

    # Data Standardization
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy()

# Predict results
def save_predictions(knn, X_test, output_file, test_ids):
    predictions = knn.predict(X_test)
    results = pd.DataFrame({'id': test_ids, 'Exited': predictions})
    results.to_csv(output_file, index=False)

# main
if __name__ == "__main__":
    # Preprocess data
    X_train, y_train, X_test = preprocess_data('train.csv', 'test.csv')

    # Get the 'id' column
    test_data = pd.read_csv('test.csv')
    test_ids = test_data['id']

    # Use the optimal parameters: n_neighbors=62, distance_metric='manhattan'
    best_knn = KNNClassifier(n_neighbors=62, distance_metric='manhattan')
    best_knn.fit(X_train, y_train)

    # Predict and save the results
    save_predictions(best_knn, X_test, 'predictions.csv', test_ids)

    # Calculate the accuracy on the training set
    y_train_pred = best_knn.predict(X_train)
    y_train_pred_labels = (y_train_pred >= 0.5).astype(int)  # 使用0.5作为阈值进行分类
    accuracy = np.mean(y_train_pred_labels == y_train)
    print("Training Accuracy: {:.2f}%".format(accuracy * 100))
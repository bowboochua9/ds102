import gzip
import numpy as np
import matplotlib.pyplot as plt

def load_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28)

def load_labels(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

X_train = load_images('train-images-idx3-ubyte.gz') / 255.0
y_train = load_labels('train-labels-idx1-ubyte.gz')
X_test  = load_images('t10k-images-idx3-ubyte.gz')  / 255.0
y_test  = load_labels('t10k-labels-idx1-ubyte.gz')


class SoftmaxRegression:
    def __init__(self, learning_rate, n_epochs, n_classes=10):
        self.lr       = learning_rate
        self.n_epochs = n_epochs
        self.K        = n_classes
        self.w        = None  
        self.b        = None  
        self.losses   = []    

    def softmax(self, z):
        z -= z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def one_hot(self, y):
        y_one_hot = np.zeros((len(y), 10))
        y_one_hot[np.arange(len(y)), y] = 1
        return y_one_hot

    def fit(self, X, y):
        N, n_features = X.shape

        self.w = np.zeros((n_features, self.K))
        self.b = np.zeros(self.K)

        y_one_hot = self.one_hot(y)

        for epoch in range(self.n_epochs):
            z     = X @ self.w + self.b       
            y_hat = self.softmax(z)          

            eps  = 1e-9  
            loss = -np.mean(
                np.sum(y_one_hot * np.log(y_hat + eps), axis=1)
            )
            self.losses.append(loss)

            diff = y_hat - y_one_hot
            dw   = (X.T @ diff) / N
            db   = diff.mean(axis=0)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1:4d}/{self.n_epochs}  |  Loss: {loss:.6f}')

        return self

    def predict_proba(self, X):
        return self.softmax(X @ self.w + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

model = SoftmaxRegression(learning_rate=0.1, n_epochs=900, n_classes=10)
model.fit(X_train, y_train)

def compute_metrics(y_true, y_pred, n_classes=10):
    precisions, recalls, f1s = [], [], []

    for k in range(n_classes):
        TP = np.sum((y_pred == k) & (y_true == k))
        FP = np.sum((y_pred == k) & (y_true != k))
        FN = np.sum((y_pred != k) & (y_true == k))
        TN = np.sum((y_pred != k) & (y_true != k))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return dict(
        accuracy  = np.mean(y_pred == y_true),
        precision = np.mean(precisions),
        recall    = np.mean(recalls),
        f1        = np.mean(f1s)
    )

    

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

train_metrics = compute_metrics(y_train, y_pred_train)
test_metrics  = compute_metrics(y_test,  y_pred_test)

print('=' * 45)
print(f'{"Metric":<12}  {"Train":>10}  {"Test":>10}')
print('=' * 45)
for key in ['accuracy', 'precision', 'recall', 'f1']:
    print(f'{key:<12}  {train_metrics[key]:>10.4f}  {test_metrics[key]:>10.4f}')
print('=' * 45)

plt.figure(figsize=(9, 4))
plt.plot(range(1, model.n_epochs + 1), model.losses, color='royalblue', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
plt.title('Loss Function trong quá trình huấn luyện', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f'Loss ban đầu : {model.losses[0]:.6f}')
print(f'Loss cuối    : {model.losses[-1]:.6f}')
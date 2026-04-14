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

X_train_full = load_images('train-images-idx3-ubyte.gz') / 255.0
y_train_full = load_labels('train-labels-idx1-ubyte.gz')
X_test_full  = load_images('t10k-images-idx3-ubyte.gz')  / 255.0
y_test_full  = load_labels('t10k-labels-idx1-ubyte.gz')


mask_train = (y_train_full == 0) | (y_train_full == 1)
mask_test  = (y_test_full  == 0) | (y_test_full  == 1)

X_train = X_train_full[mask_train]  
y_train = y_train_full[mask_train]  
X_test  = X_test_full[mask_test]
y_test  = y_test_full[mask_test]


class LogisticRegression:
    def __init__(self, learning_rate, n_epochs):
        self.lr       = learning_rate
        self.n_epochs = n_epochs
        self.w        = None  
        self.b        = None  
        self.losses   = []    

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        N, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.losses = []

        for epoch in range(self.n_epochs):
            z     = X @ self.w + self.b       
            y_hat = self.sigmoid(z)          

            eps  = 1e-9  
            loss = -np.mean(
                y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
            )
            self.losses.append(loss)

            diff = y_hat - y
            dw   = (X.T @ diff) / N
            db   = np.mean(diff)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1:4d}/{self.n_epochs}  |  Loss: {loss:.6f}')

        return self

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

model = LogisticRegression(learning_rate=0.1, n_epochs=100)
model.fit(X_train, y_train)

def compute_metrics(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (TP + TN) / len(y_true)

    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                precision=precision, recall=recall,
                f1=f1, accuracy=accuracy)

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
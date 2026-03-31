import gzip
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_images(path):
    with gzip.open(path, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

def load_labels(path):
    with gzip.open(path, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

X_train = load_images('train-images-idx3-ubyte.gz') / 255.0
y_train = load_labels('train-labels-idx1-ubyte.gz')
X_test  = load_images('t10k-images-idx3-ubyte.gz')  / 255.0
y_test  = load_labels('t10k-labels-idx1-ubyte.gz')

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    for split, X, y in [('Train', X_tr, y_tr), ('Test', X_te, y_te)]:
        p = model.predict(X)
        avg = 'binary' if len(np.unique(y)) == 2 else 'macro'
        print(f'{name} [{split}]  acc={accuracy_score(y,p):.4f}  '
              f'P={precision_score(y,p,average=avg,zero_division=0):.4f}  '
              f'R={recall_score(y,p,average=avg,zero_division=0):.4f}  '
              f'F1={f1_score(y,p,average=avg,zero_division=0):.4f}')

mask = (y_train==0)|(y_train==1)
mask_t = (y_test==0)|(y_test==1)

print('=== Logistic Regression (binary: 0 vs 1) ===')
evaluate('LogReg', LogisticRegression(max_iter=1000),
         X_train[mask], y_train[mask], X_test[mask_t], y_test[mask_t])

print('\n=== Softmax Regression (10 classes) ===')
evaluate('Softmax', LogisticRegression(solver='lbfgs', max_iter=1000),
         X_train, y_train, X_test, y_test)

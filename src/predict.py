import joblib
import numpy as np
import cv2
import sys

img = sys.argv[1]
name = sys.argv[2]
loc = sys.argv[3]

# ── 5. Predict a single image ────────────────────────────────
def predict_emotion(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48)).flatten().astype(np.float64) / 255.0
    weights = pca.transform(img.reshape(1, -1))
    label_idx = clf.predict(weights)[0]
    return le.inverse_transform([label_idx])[0]

pca = joblib.load(f'{loc}eigenface_pca_{name}.pkl')
clf = joblib.load(f'{loc}eigenface_clf_{name}.pkl')
le  = joblib.load(f'{loc}eigenface_labels_{name}.pkl')

print(predict_emotion(img))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

def cov(x, y):
    return np.mean((x - np.mean(x)) * (y - np.mean(y)))

def corr(x, y):
    return cov(x, y) / np.sqrt(np.sum((x - np.mean(x)) ** 2) * np.sum((y - np.mean(y)) ** 2))

def corr_with_target(X, y):
    cor_coef = []
    for i in range(X.shape[1]):
        a = sorted(list(zip(X[:, i], y)))
        cor_coef.append(corr([b[0] for b in a], [b[1] for b in a]))
    return np.array(cor_coef)

def inversions_ratio_metric(target, x):
    samples = len(x)
    inversions_count = 0
    target = target[np.argsort(x)]
    for i in range(samples):
        for j in range(i + 1, samples):
            if target[i] > target[j]:
                inversions_count += 1
    inversions_ratio = 2 * inversions_count / (samples * (samples - 1))
    return 1 - 2 * inversions_ratio

def monotonicity_coefficient(target, x):
    cov_target_x = cov(target, x)
    if cov_target_x == 0:
        return 0
    elif cov_target_x > 0:
        return cov_target_x / cov(sorted(target), sorted(x))
    else:
        return -cov_target_x / cov(sorted(target), sorted(x, key=lambda x: -x))

class DetectorBasedClassifier(object):
    def __init__(self, features_mask):
        self.mask = features_mask

    def fit(self, responders, nonresponders):
        self.mask = features_mask
        self.responders = (responders * self.mask.reshape(1, -1))[:, self.mask != 0]
        self.nonresponders = (nonresponders * self.mask.reshape(1, -1))[:, self.mask != 0]
        return self

    def detect(self, x):
        positiveness = np.mean(x > self.responders) + 1e-20
        negativeness = np.mean(x < self.nonresponders) + 1e-20
        norm = positiveness + negativeness
        return negativeness / norm, positiveness / norm

    def predict_proba(self, X):
        X = (X * self.mask.reshape(1, -1))[:, self.mask != 0]
        return np.apply_along_axis(arr=X, axis=1, func1d=self.detect)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > proba[:, 0]).astype(float)

class WeightedDetectorBasedClassifier(DetectorBasedClassifier):
    def __init__(self, features_mask, features_weights):
        super().__init__(features_mask)
        self.features_weights = features_weights.reshape(1, -1)

    def fit(self, responders, nonresponders):
        self.responders = (responders * self.mask.reshape(1, -1))[:, self.mask != 0]
        self.nonresponders = (nonresponders * self.mask.reshape(1, -1))[:, self.mask != 0]
        return self

    def detect(self, x):
        positiveness = ((x > self.responders) * self.features_weights).mean() + 1e-20
        negativeness = ((x < self.nonresponders) * self.features_weights).mean() + 1e-20
        norm = positiveness + negativeness
        return negativeness / norm, positiveness / norm
    
def convert_to_cell_lines_pc(cell_data, patients_data, features_mask, monotonicity=1):
    scaler = StandardScaler().fit(cell_data[:, features_mask == monotonicity])
    pca_transformer = PCA(n_components=(features_mask == monotonicity).sum())
    pca_transformer.fit(cell_data[:, features_mask == monotonicity])
    cell_reduced = pca_transformer.transform(
                       scaler.transform(cell_data[:, features_mask == monotonicity]))
    pat_reduced = pca_transformer.transform(
                      scaler.transform(patients_data[:, features_mask == monotonicity]))
    return cell_reduced, pat_reduced

def evaluate_auc(X_patients, y_patients, X_cell_lines, y_cell_lines):
    
    corr_coefs = corr_with_target(X_cell_lines, y_cell_lines)

    features_mask = (corr_coefs > 0).astype(float) - (corr_coefs < 0).astype(float)
    
    cell_inc_reduced, pat_inc_reduced = convert_to_cell_lines_pc(X_cell_lines, X_patients,
                                                                 features_mask, 1)
    cell_dec_reduced, pat_dec_reduced = convert_to_cell_lines_pc(X_cell_lines, X_patients,
                                                                 features_mask, -1)

    X_pat_reduced = np.hstack([pat_inc_reduced, pat_dec_reduced])
    X_cell_reduced = np.hstack([cell_inc_reduced, cell_dec_reduced])

    corr_coefs_reduced = corr_with_target(X_cell_reduced, y_cell_lines)
    features_mask_reduced = (corr_coefs_reduced > 0).astype(float) - (corr_coefs_reduced < 0).astype(float)
    features_weights_reduced = np.abs(corr_coefs_reduced)

    roc = []
    
    clf = WeightedDetectorBasedClassifier(features_mask_reduced, features_weights_reduced)
    for i in range(1):
        prediction = np.zeros((X_patients.shape[0], 2))
        cv = LeaveOneOut().split(X_pat_reduced)
        for train, test in cv:
            clf.fit(X_pat_reduced[train][y_patients[train]==100],
                    X_pat_reduced[train][y_patients[train]==0])
            prediction[test] = clf.predict_proba(X_pat_reduced[test])
        roc.append(roc_auc_score(y_patients/100, prediction[:, 1]))
    return np.mean(roc)

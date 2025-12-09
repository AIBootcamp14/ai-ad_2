import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

class Evaluator:
    def __init__(self, high_pct = 70, low_pct = 70):
        csv_path = DATA_DIR / 'pseudo_label.csv'
        self.score = pd.read_csv(csv_path).anomaly_score.values
        self.high_pct = high_pct
        self.low_pct = low_pct
        pseudo_y, _, _ = self.create_pseudo_labels()
        self.pseudo_y = pseudo_y

    def create_pseudo_labels(self):
        high_th = np.percentile(self.score, self.high_pct)
        low_th = np.percentile(self.score, self.low_pct)

        pseudo_y = np.full_like(self.score, fill_value = -1, dtype = int)
        pseudo_y[self.score >= high_th] = 1
        pseudo_y[self.score < low_th] = 0

        return pseudo_y, low_th, high_th

    def evaluate(self, pred):
        f1 = f1_score(pred, pd.Series(self.pseudo_y), zero_division = 0)
        return f1

if __name__ == '__main__':
    eval = Evaluator()
    pred = pd.read_csv(PROJECT_ROOT / 'output_hydra_ee.csv')['faultNumber']
    pred2 = pd.read_csv(PROJECT_ROOT / 'output_hydra_if.csv')['faultNumber']
    f1 = eval.evaluate(pred)
    f1_2 = eval.evaluate(pred2)

    print(f"sample 1 [ee] f1 score : {f1}")
    print(f"sample 2 [if] f1 score : {f1_2}")
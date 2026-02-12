import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple

from model_utils import (
    load_and_prepare_data,
    create_price_feature_frame,
    WalkForwardValidator,
    calculate_metrics,
)

ASSET_PATH = Path("asset_events_df.csv")
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

HORIZON = 5
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
LOG_RET_CLIP = 0.35
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


def build_sequences(features: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    indices = []
    for idx in range(seq_len - 1, len(features)):
        window = features[idx - seq_len + 1 : idx + 1]
        sequences.append(window)
        indices.append(idx)
    return np.array(sequences), np.array(indices)


def compute_norm_stats(train_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_seq.mean(axis=(0, 1), keepdims=True)
    std = train_seq.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1e-6, std)
    return mean, std


def normalize_with_stats(seq: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (seq - mean) / std


def train_one_split(model, train_loader):
    criterion = nn.HuberLoss(delta=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


def run_lstm_analysis():
    print("Loading data...")
    df = load_and_prepare_data(ASSET_PATH)
    companies = df['company'].unique()
    results = []

    validator = WalkForwardValidator(n_splits=5)

    print(f"Starting LSTM analysis for {len(companies)} companies...")
    for i, company in enumerate(companies):
        if i % 5 == 0:
            print(f"Processing company {i+1}/{len(companies)}: {company}")

        company_df = df[df['company'] == company].sort_values('date').reset_index(drop=True)
        model_data, feature_cols, target_cols = create_price_feature_frame(
            company_df,
            horizon=HORIZON,
            n_price_lags=20,
            n_return_lags=15,
        )

        if len(model_data) < SEQ_LEN + 10:
            continue

        feature_matrix = model_data[feature_cols].values.astype(np.float32)
        target_price_matrix = model_data[target_cols].values.astype(np.float32)
        ref_prices = model_data['close'].values.astype(np.float32)

        sequences, seq_indices = build_sequences(feature_matrix, SEQ_LEN)
        target_indices = seq_indices
        log_returns = np.log(target_price_matrix / ref_prices[:, None])
        target_returns = log_returns[target_indices]
        ref_price_seq = ref_prices[target_indices]

        if len(sequences) < 100:
            continue

        for split_idx, (train_idx, test_idx) in enumerate(validator.split(sequences)):
            if len(test_idx) == 0 or len(train_idx) < 50:
                continue

            train_seq = sequences[train_idx]
            test_seq = sequences[test_idx]
            mean, std = compute_norm_stats(train_seq)
            train_seq_norm = normalize_with_stats(train_seq, mean, std)
            test_seq_norm = normalize_with_stats(test_seq, mean, std)

            train_returns = target_returns[train_idx]
            test_returns = target_returns[test_idx]
            test_ref_prices = ref_price_seq[test_idx]
            test_targets_prices = target_price_matrix[target_indices[test_idx]]

            train_dataset = SequenceDataset(train_seq_norm, train_returns)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            model = LSTMForecaster(
                input_size=train_seq.shape[-1],
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
                output_size=HORIZON,
            ).to(DEVICE)

            train_one_split(model, train_loader)

            model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(test_seq_norm, dtype=torch.float32, device=DEVICE)
                pred_returns = model(test_tensor).cpu().numpy()

            pred_returns = np.clip(pred_returns, -LOG_RET_CLIP, LOG_RET_CLIP)
            pred_prices = test_ref_prices[:, None] * np.exp(pred_returns)

            for j in range(len(test_idx)):
                metrics = calculate_metrics(test_targets_prices[j], pred_prices[j])
                metrics['company'] = company
                metrics['split'] = split_idx
                results.append(metrics)

    if results:
        results_df = pd.DataFrame(results)
        output_path = OUTPUT_DIR / "lstm_results.csv"
        results_df.to_csv(output_path, index=False)

        company_summary = (
            results_df.groupby('company')[['mae', 'rmse', 'mape', 'r2']]
            .mean()
            .reset_index()
            .sort_values('rmse', ascending=False)
        )
        summary_path = OUTPUT_DIR / "lstm_company_summary.csv"
        company_summary.to_csv(summary_path, index=False)

        avg_metrics = results_df[['mae', 'rmse', 'mape', 'r2']].mean()
        print("\nAverage LSTM Performance (on Prices):")
        print(avg_metrics)
        print("\nPer-company summary (worst RMSE first):")
        print(company_summary)
        print(f"\nDetailed results saved to {output_path}")
        print(f"Company summary saved to {summary_path}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    run_lstm_analysis()

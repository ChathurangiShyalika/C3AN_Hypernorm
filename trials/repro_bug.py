import pandas as pd
from model_utils import load_and_prepare_data
from pathlib import Path

ASSET_PATH = Path("asset_events_df.csv")

def check_data():
    print("Loading data...")
    df = load_and_prepare_data(ASSET_PATH)
    
    company = "Alphabet Inc."
    print(f"Checking {company}...")
    
    company_df = df[df['company'] == company].sort_values('date').reset_index(drop=True)
    
    print("First 10 rows of close:")
    print(company_df[['date', 'close']].head(10))
    
    # Check for duplicates
    dups = company_df[company_df.duplicated(subset=['date'], keep=False)]
    if not dups.empty:
        print("Duplicates found:")
        print(dups)
    else:
        print("No duplicates found.")
        
    # Apply shifts
    for h in range(1, 6):
        company_df[f'target_close_h{h}'] = company_df['close'].shift(-h)
        
    print("\nFirst 5 rows with targets:")
    print(company_df[['date', 'close'] + [f'target_close_h{h}' for h in range(1, 6)]].head(5))
    
    # Check zero std
    target_cols = [f'target_close_h{h}' for h in range(1, 6)]
    zero_std = (company_df[target_cols].std(axis=1) == 0)
    print(f"\nFraction zero std: {zero_std.mean():.3%}")
    
    if zero_std.any():
        print("Problematic rows:")
        print(company_df[zero_std][['date', 'close'] + target_cols].head(5))

if __name__ == "__main__":
    check_data()

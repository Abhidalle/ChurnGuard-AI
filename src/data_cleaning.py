import pandas as pd
import os

# 1. INPUT/OUTPUT
# We take the big file you just made and create a clean master version
INPUT_FILE = 'data/intermediate/nepal_birds.csv'
OUTPUT_FILE = 'data/processed/nepal_master_dataset.csv'

def create_better_version():
    print(f"🛠️ Refining {INPUT_FILE} into the AI-Ready version...")

    # 2. SELECT THE "GOLDEN COLUMNS"
    # We only keep the 6 columns that actually help predict birds
    target_cols = [
        'COMMON NAME', 'OBSERVATION COUNT', 'LATITUDE', 
        'LONGITUDE', 'OBSERVATION DATE', 'LOCALITY'
    ]
    
    # Load the data - we use usecols to ignore the 40+ useless columns
    df = pd.read_csv(INPUT_FILE, usecols=target_cols)

    # 3. THE 2000-2026 FILTER
    # Convert text to dates and drop anything before the year 2000
    df['OBSERVATION DATE'] = pd.to_datetime(df['OBSERVATION DATE'], errors='coerce')
    df = df.dropna(subset=['OBSERVATION DATE'])
    df = df[df['OBSERVATION DATE'].dt.year >= 2000]

    # 4. CLEAN NAMES
    # Change 'COMMON NAME' to 'common_name' (Python likes lowercase)
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    # 5. SAVE THE "BETTER" VERSION
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("-" * 30)
    print(f"✅ BETTER VERSION CREATED!")
    print(f"📊 Rows remaining: {len(df):,}")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    create_better_version()
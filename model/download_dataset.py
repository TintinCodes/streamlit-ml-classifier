"""
Download Bank Marketing dataset from Kaggle using kagglehub
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def download_and_prepare_dataset():
    """Download dataset from Kaggle and prepare 550 balanced samples"""
    
    print("="*60)
    print("Downloading Bank Marketing Dataset from Kaggle")
    print("="*60)
    
    try:
        # Import kagglehub
        import kagglehub
        
        print("\n1. Downloading dataset from Kaggle...")
        
        # Download the dataset (this downloads all files to a local folder)
        path = kagglehub.dataset_download("janiobachmann/bank-marketing-dataset")
        
        print(f"✓ Dataset downloaded to: {path}")
        
        # List all files in the downloaded folder
        files = os.listdir(path)
        print(f"✓ Available files: {files}")
        
        # Find the CSV file (usually bank.csv or bank-full.csv)
        csv_file = None
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(path, file)
                print(f"✓ Found CSV file: {file}")
                break
        
        if csv_file is None:
            raise FileNotFoundError("No CSV file found in downloaded dataset")
        
        # Load the CSV file - try different separators
        df = None
        separators = [';', ',', '\t']
        
        for sep in separators:
            try:
                df_test = pd.read_csv(csv_file, sep=sep, nrows=5)
                # Check if we got multiple columns (not all merged into one)
                if len(df_test.columns) > 5:
                    df = pd.read_csv(csv_file, sep=sep)
                    print(f"✓ Loaded with separator: '{sep}'")
                    break
            except:
                continue
        
        if df is None:
            raise ValueError("Could not parse CSV file with any separator")
        
        print(f"✓ Original shape: {df.shape}")
        print(f"✓ Columns ({len(df.columns)}): {list(df.columns)[:5]}...")
        
        # Check what the target column is named
        if 'y' in df.columns:
            target_col = 'y'
        elif 'deposit' in df.columns:
            target_col = 'deposit'
        else:
            # Print all columns to help debug
            print(f"\nAll columns: {list(df.columns)}")
            raise ValueError("Cannot find target column (y or deposit)")
        
        print(f"\n2. Checking class distribution...")
        print(f"Target column: '{target_col}'")
        print(df[target_col].value_counts())
        
        # Check if we have both classes
        unique_values = df[target_col].unique()
        print(f"Unique values in target: {unique_values}")
        
        # Determine positive and negative class labels
        if 'yes' in unique_values and 'no' in unique_values:
            positive_label = 'yes'
            negative_label = 'no'
        elif 1 in unique_values and 0 in unique_values:
            positive_label = 1
            negative_label = 0
        else:
            raise ValueError(f"Unexpected target values: {unique_values}")
        
        # Count samples per class
        positive_count = (df[target_col] == positive_label).sum()
        negative_count = (df[target_col] == negative_label).sum()
        
        print(f"\nClass distribution:")
        print(f"  Positive ({positive_label}): {positive_count}")
        print(f"  Negative ({negative_label}): {negative_count}")
        
        # Check if we have enough samples
        if positive_count < 275 or negative_count < 275:
            print(f"\nWARNING: Not enough samples for balanced 275/275 split!")
            print(f"Will use all available samples from minority class")
            sample_size = min(positive_count, negative_count)
        else:
            sample_size = 275
        
        # Sample from each class
        print(f"\n3. Sampling {sample_size*2} instances ({sample_size} per class)...")
        
        df_positive = df[df[target_col] == positive_label].sample(
            n=sample_size, 
            random_state=42
        )
        df_negative = df[df[target_col] == negative_label].sample(
            n=sample_size, 
            random_state=42
        )
        
        # Combine and shuffle
        df_sampled = pd.concat([df_positive, df_negative])
        df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✓ Sampled to {len(df_sampled)} instances")
        print(f"\nNew distribution:")
        print(df_sampled[target_col].value_counts())
        
        # Rename target column to 'deposit' for consistency
        if target_col != 'deposit':
            df_sampled = df_sampled.rename(columns={target_col: 'deposit'})
        
        # Save to model folder
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, 'bank_marketing.csv')
        
        df_sampled.to_csv(output_path, index=False)
        
        print(f"\n4. Saving dataset...")
        print(f"✓ Saved as: {output_path}")
        print(f"✓ Shape: {df_sampled.shape}")
        print(f"✓ Features: {len(df_sampled.columns) - 1}")
        
        print("\n" + "="*60)
        print("✓ SUCCESS! Ready to train models!")
        print("="*60)
        print(f"\nNext step: python model/train_models.py")
        
        return df_sampled
        
    except ImportError:
        print("\n" + "="*60)
        print("ERROR: kagglehub not installed")
        print("="*60)
        print("\nPlease install it first:")
        print("  pip install kagglehub")
        print("\nThen run this script again.")
        return None
        
    except Exception as e:
        import traceback
        print(f"\n" + "="*60)
        print(f"ERROR: {e}")
        print("="*60)
        traceback.print_exc()
        print("\nAlternative: Manual Download")
        print("1. Go to: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset")
        print("2. Download 'bank-full.csv' or 'bank.csv'")
        print("3. Save it in model/ folder")
        print("4. Run this script again (it will auto-detect the file)")
        return None

def process_manual_dataset():
    """Process manually downloaded dataset"""
    
    print("\nProcessing manually downloaded dataset...")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible filenames
    possible_files = [
        os.path.join(script_dir, 'bank-full.csv'),
        os.path.join(script_dir, 'bank.csv'),
        os.path.join(script_dir, 'bank_marketing_manual.csv'),
        os.path.join(script_dir, 'bank-additional-full.csv'),
    ]
    
    df = None
    loaded_file = None
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            # Try different separators
            for sep in [';', ',', '\t']:
                try:
                    df_test = pd.read_csv(filepath, sep=sep, nrows=5)
                    # Check if we got multiple columns
                    if len(df_test.columns) > 5:
                        df = pd.read_csv(filepath, sep=sep)
                        loaded_file = filepath
                        print(f"✓ Loaded: {os.path.basename(filepath)} (separator: '{sep}')")
                        break
                except:
                    continue
            if df is not None:
                break
    
    if df is None:
        print("ERROR: Could not find or parse manual file!")
        print("\nPlease download from Kaggle:")
        print("https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset")
        print("\nAnd save one of these files in the model/ folder:")
        for f in possible_files:
            print(f"  - {os.path.basename(f)}")
        return None
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:5]}...")
    
    # Find target column
    if 'y' in df.columns:
        target_col = 'y'
    elif 'deposit' in df.columns:
        target_col = 'deposit'
    else:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Cannot find target column")
    
    print(f"Target distribution:\n{df[target_col].value_counts()}")
    
    # Determine class labels
    unique_values = df[target_col].unique()
    if 'yes' in unique_values and 'no' in unique_values:
        positive_label = 'yes'
        negative_label = 'no'
    elif 1 in unique_values and 0 in unique_values:
        positive_label = 1
        negative_label = 0
    else:
        raise ValueError(f"Unexpected target values: {unique_values}")
    
    # Sample balanced dataset (275 each = 550 total)
    positive_count = (df[target_col] == positive_label).sum()
    negative_count = (df[target_col] == negative_label).sum()
    sample_size = min(275, positive_count, negative_count)
    
    print(f"\nSampling {sample_size} from each class...")
    
    df_positive = df[df[target_col] == positive_label].sample(n=sample_size, random_state=42)
    df_negative = df[df[target_col] == negative_label].sample(n=sample_size, random_state=42)
    
    df_sampled = pd.concat([df_positive, df_negative])
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Rename target to 'deposit'
    if target_col != 'deposit':
        df_sampled = df_sampled.rename(columns={target_col: 'deposit'})
    
    # Save
    output_path = os.path.join(script_dir, 'bank_marketing.csv')
    df_sampled.to_csv(output_path, index=False)
    
    print(f"\n✓ Processed successfully!")
    print(f"✓ Saved as: {output_path}")
    print(f"✓ Shape: {df_sampled.shape}")
    print(f"New distribution:\n{df_sampled['deposit'].value_counts()}")
    
    return df_sampled

if __name__ == "__main__":
    # Try Kaggle download first
    df = download_and_prepare_dataset()
    
    # If failed, try manual processing
    if df is None:
        df = process_manual_dataset()
    
    if df is not None:
        print("\n✓ Dataset ready! Run: python model/train_models.py")
    else:
        print("\n✗ Failed to prepare dataset. Please download manually.")
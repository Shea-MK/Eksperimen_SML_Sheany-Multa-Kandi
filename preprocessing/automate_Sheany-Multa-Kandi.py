import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):
    """Memuat dataset College dari path yang ditentukan."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan!")
    return pd.read_csv(file_path).copy()

def process_college_data(df):
    """Menjalankan seluruh alur preprocessing untuk dataset College."""
    
    # 1. HANDLING DUPLICATES & MISSING VALUES
    df.drop_duplicates(inplace=True)
    
    # Mengisi nilai numerik yang kosong dengan median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 2. HANDLING OUTLIERS (IQR Method)
    # Menghapus data ekstrem agar tidak merusak hasil scaling
    outlier_cols = ['Apps', 'Accept', 'Enroll', 'Expend']
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # 3. FEATURE ENGINEERING
    # Membuat fitur Accept_Rate (Persentase pendaftar yang diterima)
    if 'Accept' in df.columns and 'Apps' in df.columns:
        df['Accept_Rate'] = (df['Accept'] / df['Apps']) * 100

    # 4. DATA BINNING
    # Mengelompokkan universitas berdasarkan Graduation Rate
    if 'Grad.Rate' in df.columns:
        labels = ['Low_Grad_Rate', 'Medium_Grad_Rate', 'High_Grad_Rate']
        df['Grad_Category'] = pd.qcut(df['Grad.Rate'], q=3, labels=labels)

    # 5. SCALING (Penskalaan Fitur)
    # Standarisasi (StandardScaler) untuk fitur dengan variansi besar
    std_scaler = StandardScaler()
    cols_to_std = ['Apps', 'Accept', 'Enroll', 'Outstate', 'Expend', 'Room.Board']
    cols_to_std = [c for c in cols_to_std if c in df.columns]
    df[cols_to_std] = std_scaler.fit_transform(df[cols_to_std])

    # Normalisasi (MinMaxScaler) untuk persentase/rasio (0-1)
    mm_scaler = MinMaxScaler()
    cols_to_norm = ['PhD', 'S.F.Ratio', 'Grad.Rate', 'Accept_Rate', 'perc.alumni']
    cols_to_norm = [c for c in cols_to_norm if c in df.columns]
    df[cols_to_norm] = mm_scaler.fit_transform(df[cols_to_norm])

    # 6. ENCODING
    # Mengubah fitur kategorikal menjadi angka (0 dan 1)
    categorical_cols = ['Private', 'Grad_Category']
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    # Menggunakan dtype=int agar hasil dummy adalah 0/1 (bukan boolean)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    return df

def save_preprocessed_data(df, output_path):
    """Menyimpan hasil akhir ke file CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sukses! Data telah diproses dan disimpan di: {output_path}")

if __name__ == "__main__":
    # Konfigurasi Path sesuai kriteria folder submission
    INPUT_PATH = "namadataset_raw/College.csv"
    OUTPUT_PATH = "preprocessing/College_preprocessed.csv"

    try:
        print("Memulai proses otomatisasi untuk Sheany_Multa_Kandi...")
        college_df = load_data(INPUT_PATH)
        processed_df = process_college_data(college_df)
        save_preprocessed_data(processed_df, OUTPUT_PATH)
        print("Seluruh tahapan selesai dengan sukses.")
    except Exception as e:
        print(f"Gagal menjalankan otomatisasi: {e}")
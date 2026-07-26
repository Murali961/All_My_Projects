# ============================================================
#  FULL DATA PREPROCESSING PIPELINE
#  Dataset: Indian Pharmaceutical Products
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_csv('indian_pharmaceutical_products_clean.csv')

print(f"Shape           : {df.shape}")
print(f"Rows            : {df.shape[0]:,}")
print(f"Columns         : {df.shape[1]}")
print(f"\nColumn Names    : {list(df.columns)}")


# ============================================================
# STEP 2: DATA INSPECTION / EDA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: DATA INSPECTION / EDA")
print("=" * 60)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Info ---")
df.info()

print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

print("\n--- Unique Value Counts (Categorical) ---")
cat_cols = ['dosage_form', 'pack_unit', 'therapeutic_class', 'manufacturer']
for col in cat_cols:
    print(f"\n{col} ({df[col].nunique()} unique):")
    print(df[col].value_counts().head(10))

print("\n--- Boolean Column ---")
print(df['is_discontinued'].value_counts())

# ── EDA Plots ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('EDA: Indian Pharmaceutical Products', fontsize=16)

# Price distribution
axes[0, 0].hist(df['price_inr'].clip(upper=1000), bins=50, color='steelblue', edgecolor='white')
axes[0, 0].set_title('Price Distribution (INR, capped at 1000)')
axes[0, 0].set_xlabel('Price (INR)')

# Dosage form counts
df['dosage_form'].value_counts().head(10).plot(kind='bar', ax=axes[0, 1], color='coral')
axes[0, 1].set_title('Top 10 Dosage Forms')
axes[0, 1].tick_params(axis='x', rotation=45)

# Therapeutic class
df['therapeutic_class'].value_counts().head(10).plot(kind='bar', ax=axes[0, 2], color='mediumseagreen')
axes[0, 2].set_title('Therapeutic Classes')
axes[0, 2].tick_params(axis='x', rotation=45)

# Pack size distribution
axes[1, 0].hist(df['pack_size'].dropna().clip(upper=100), bins=40, color='mediumpurple', edgecolor='white')
axes[1, 0].set_title('Pack Size Distribution (capped at 100)')

# Num active ingredients
df['num_active_ingredients'].value_counts().sort_index().plot(
    kind='bar', ax=axes[1, 1], color='goldenrod')
axes[1, 1].set_title('Number of Active Ingredients')

# Missing values heatmap
missing_cols = df.columns[df.isnull().any()].tolist()
if missing_cols:
    sns.heatmap(df[missing_cols].isnull().head(200), ax=axes[1, 2],
                cbar=False, cmap='viridis', yticklabels=False)
    axes[1, 2].set_title('Missing Values (first 200 rows)')
else:
    axes[1, 2].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=12)
    axes[1, 2].set_title('Missing Values')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nEDA plots saved to: eda_plots.png")


# ============================================================
# STEP 3: HANDLE DUPLICATES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: HANDLE DUPLICATES")
print("=" * 60)

before = len(df)
print(f"Duplicate rows   : {df.duplicated().sum()}")
print(f"Duplicate product_id: {df['product_id'].duplicated().sum()}")

df.drop_duplicates(inplace=True)
print(f"Rows before      : {before:,}")
print(f"Rows after       : {len(df):,}")
print(f"Removed          : {before - len(df)}")


# ============================================================
# STEP 4: DATA TYPE CONVERSION
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: DATA TYPE CONVERSION")
print("=" * 60)

# is_discontinued is already bool — confirm
print(f"is_discontinued dtype: {df['is_discontinued'].dtype}")

# product_id as int (confirm)
df['product_id'] = df['product_id'].astype(int)

# price_inr, pack_size as float (confirm)
df['price_inr'] = df['price_inr'].astype(float)
df['pack_size']  = df['pack_size'].astype(float)

# Ensure string cols are clean strings
str_cols = ['brand_name', 'manufacturer', 'dosage_form', 'pack_unit',
            'primary_ingredient', 'primary_strength', 'therapeutic_class',
            'packaging_raw', 'manufacturer_raw']
for col in str_cols:
    df[col] = df[col].astype(str).str.strip()

print("All data types confirmed and converted.")
print(df.dtypes)


# ============================================================
# STEP 5: CLEAN STRING COLUMNS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: CLEAN STRING COLUMNS")
print("=" * 60)

# Fix double spaces and inconsistent casing in manufacturer
df['manufacturer'] = (
    df['manufacturer']
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)   # collapse double spaces
)

df['manufacturer_raw'] = (
    df['manufacturer_raw']
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
)

# Lowercase dosage_form, therapeutic_class, pack_unit for consistency
df['dosage_form']       = df['dosage_form'].str.lower().str.strip()
df['therapeutic_class'] = df['therapeutic_class'].str.lower().str.strip()
df['pack_unit']         = df['pack_unit'].str.lower().str.strip()

print("Cleaned: manufacturer, dosage_form, therapeutic_class, pack_unit")
print("\nSample manufacturer values:")
print(df['manufacturer'].value_counts().head(5))


# ============================================================
# STEP 6: HANDLE INVALID PRICES (price = 0)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: HANDLE INVALID PRICES")
print("=" * 60)

zero_price = (df['price_inr'] == 0).sum()
print(f"Rows with price_inr = 0: {zero_price}")

# Drop rows with price = 0 (not free drugs — data error)
df = df[df['price_inr'] > 0].copy()
print(f"Dropped {zero_price} invalid rows. New shape: {df.shape}")


# ============================================================
# STEP 7: HANDLE MISSING VALUES
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: HANDLE MISSING VALUES")
print("=" * 60)

# --- 7a. pack_size: fill with median per dosage_form ---
missing_pack_size = df['pack_size'].isnull().sum()
print(f"Missing pack_size     : {missing_pack_size:,}")
df['pack_size'] = df.groupby('dosage_form')['pack_size'].transform(
    lambda x: x.fillna(x.median())
)
# fallback: global median for any remaining
df['pack_size'] = df['pack_size'].fillna(df['pack_size'].median())
print(f"pack_size nulls after : {df['pack_size'].isnull().sum()}")

# --- 7b. pack_unit: fill with mode per dosage_form ---
missing_pack_unit = df['pack_unit'].isnull().sum()
print(f"\nMissing pack_unit     : {missing_pack_unit:,}")
df['pack_unit'] = df.groupby('dosage_form')['pack_unit'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'unknown')
)
df['pack_unit'] = df['pack_unit'].replace('nan', 'unknown').fillna('unknown')
print(f"pack_unit nulls after : {df['pack_unit'].isnull().sum()}")

# --- 7c. primary_strength: fill with 'unknown' (cannot infer strength) ---
missing_strength = df['primary_strength'].isnull().sum()
print(f"\nMissing primary_strength : {missing_strength:,}")
df['primary_strength'] = df['primary_strength'].replace('nan', np.nan)
df['primary_strength'] = df['primary_strength'].fillna('unknown')
print(f"primary_strength nulls after: {df['primary_strength'].isnull().sum()}")

print(f"\nAll missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print("✅ No missing values remain." if df.isnull().sum().sum() == 0 else "")


# ============================================================
# STEP 8: HANDLE OUTLIERS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: HANDLE OUTLIERS")
print("=" * 60)

def detect_outliers_iqr(series, multiplier=3.0):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return lower, upper

# ── price_inr ──────────────────────────────────────────────
lower_p, upper_p = detect_outliers_iqr(df['price_inr'])
outliers_p = ((df['price_inr'] < lower_p) | (df['price_inr'] > upper_p)).sum()
print(f"price_inr  → lower: {lower_p:.2f}, upper: {upper_p:.2f}, outliers: {outliers_p:,}")
# Cap (Winsorize): keeps all rows, clips extreme values
df['price_inr_capped'] = df['price_inr'].clip(lower=max(0, lower_p), upper=upper_p)

# ── pack_size ──────────────────────────────────────────────
lower_s, upper_s = detect_outliers_iqr(df['pack_size'])
outliers_s = ((df['pack_size'] < lower_s) | (df['pack_size'] > upper_s)).sum()
print(f"pack_size  → lower: {lower_s:.2f}, upper: {upper_s:.2f}, outliers: {outliers_s:,}")
df['pack_size_capped'] = df['pack_size'].clip(lower=max(1, lower_s), upper=upper_s)

# ── Outlier visualization ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Outlier Detection: Before vs After Capping', fontsize=14)
axes[0].boxplot(df['price_inr'].clip(upper=5000), vert=True)
axes[0].set_title('price_inr (original)')
axes[1].boxplot(df['price_inr_capped'], vert=True)
axes[1].set_title('price_inr_capped')
plt.tight_layout()
plt.savefig('outlier_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Outlier plots saved to: outlier_plots.png")


# ============================================================
# STEP 9: ENCODE CATEGORICAL VARIABLES
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: ENCODE CATEGORICAL VARIABLES")
print("=" * 60)

# ── 9a. Label Encoding (ordinal-like / for tree models) ───
le_cols = ['dosage_form', 'pack_unit', 'therapeutic_class']
label_encoders = {}
for col in le_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Label encoded '{col}' → {col}_encoded  | classes: {list(le.classes_)}")

# ── 9b. One-Hot Encoding (for ML models needing dummies) ──
df_ohe = pd.get_dummies(df[['dosage_form', 'pack_unit', 'therapeutic_class']],
                         prefix=['df', 'pu', 'tc'],
                         drop_first=True)
print(f"\nOne-Hot Encoding produced {df_ohe.shape[1]} dummy columns")
# Attach (optional — useful for logistic regression / linear models)
df_with_ohe = pd.concat([df.reset_index(drop=True), df_ohe.reset_index(drop=True)], axis=1)

# ── 9c. Boolean → int ─────────────────────────────────────
df['is_discontinued_int'] = df['is_discontinued'].astype(int)
print(f"\nis_discontinued → is_discontinued_int (0/1)")
print(df['is_discontinued_int'].value_counts())


# ============================================================
# STEP 10: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: FEATURE ENGINEERING")
print("=" * 60)

# Price per unit (cost efficiency)
df['price_per_unit'] = (df['price_inr'] / df['pack_size']).round(4)

# Capped price per unit
df['price_per_unit_capped'] = (df['price_inr_capped'] / df['pack_size_capped']).round(4)

# Is combination drug (more than 1 ingredient)
df['is_combination'] = (df['num_active_ingredients'] > 1).astype(int)

# Brand name length (proxy for complexity / marketing)
df['brand_name_length'] = df['brand_name'].str.len()

# Brand name word count
df['brand_name_words'] = df['brand_name'].str.split().str.len()

# Manufacturer is a large pharma (top 10 by volume)
top10_mfr = df['manufacturer'].value_counts().head(10).index.tolist()
df['is_top_manufacturer'] = df['manufacturer'].isin(top10_mfr).astype(int)

# Price tier bucket
df['price_tier'] = pd.cut(
    df['price_inr'],
    bins=[0, 50, 150, 500, 5000, df['price_inr'].max()],
    labels=['budget', 'low', 'mid', 'premium', 'ultra'],
    right=True
)
df['price_tier_encoded'] = LabelEncoder().fit_transform(df['price_tier'].astype(str))

print("New features created:")
new_feats = ['price_per_unit', 'price_per_unit_capped', 'is_combination',
             'brand_name_length', 'brand_name_words', 'is_top_manufacturer',
             'price_tier', 'price_tier_encoded']
for f in new_feats:
    print(f"  ✔ {f}")

print(f"\nSample:\n{df[new_feats].head()}")


# ============================================================
# STEP 11: DROP REDUNDANT COLUMNS
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: DROP REDUNDANT COLUMNS")
print("=" * 60)

# packaging_raw and manufacturer_raw are duplicates of cleaner cols
cols_to_drop = ['packaging_raw', 'manufacturer_raw']
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped: {cols_to_drop}")
print(f"Remaining columns ({len(df.columns)}): {list(df.columns)}")


# ============================================================
# STEP 12: FEATURE SCALING
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: FEATURE SCALING")
print("=" * 60)

numeric_features = ['price_inr_capped', 'pack_size_capped',
                    'num_active_ingredients', 'price_per_unit_capped',
                    'brand_name_length']

# Standard Scaler (Z-score) — best for normally-ish distributed
scaler_std = StandardScaler()
df_std_scaled = df[numeric_features].copy()
df_std_scaled[numeric_features] = scaler_std.fit_transform(df[numeric_features])
print("StandardScaler applied:")
print(df_std_scaled[numeric_features].describe().round(3))

# MinMax Scaler (0–1) — best for neural nets / distance-based
scaler_mm = MinMaxScaler()
df_mm_scaled = df[numeric_features].copy()
df_mm_scaled[numeric_features] = scaler_mm.fit_transform(df[numeric_features])
print("\nMinMaxScaler applied (range 0–1):")
print(df_mm_scaled[numeric_features].describe().round(3))

# Add scaled cols to main df
for col in numeric_features:
    df[f'{col}_std'] = df_std_scaled[col]
    df[f'{col}_minmax'] = df_mm_scaled[col]
print("\nScaled columns added with _std and _minmax suffixes")


# ============================================================
# STEP 13: TRAIN / TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 13: TRAIN / TEST SPLIT")
print("=" * 60)

# Define features and target (predicting price tier as example)
feature_cols = [
    'pack_size_capped', 'num_active_ingredients', 'is_combination',
    'is_discontinued_int', 'brand_name_length', 'brand_name_words',
    'is_top_manufacturer', 'dosage_form_encoded',
    'pack_unit_encoded', 'therapeutic_class_encoded'
]
target_col = 'price_tier_encoded'

X = df[feature_cols]
y = df[target_col]

# Stratified split (preserves class ratios)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Further split train → train + validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"Train set   : {X_train.shape[0]:,} rows ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Val set     : {X_val.shape[0]:,} rows  ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"Test set    : {X_test.shape[0]:,} rows  ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"\nTarget distribution (train):\n{y_train.value_counts()}")


# ============================================================
# STEP 14: FULL SKLEARN PIPELINE (Production Ready)
# ============================================================
print("\n" + "=" * 60)
print("STEP 14: SKLEARN PIPELINE (PRODUCTION READY)")
print("=" * 60)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Define column groups for raw data pipeline
numeric_cols  = ['pack_size', 'num_active_ingredients',
                 'price_inr', 'brand_name_length']
categorical_cols = ['dosage_form', 'pack_unit', 'therapeutic_class']

# Numeric sub-pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

# Categorical sub-pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combined preprocessor
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
], remainder='drop')

# Full pipeline with model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    ))
])

# --- Prepare raw feature data for pipeline ---
raw_feature_cols = numeric_cols + categorical_cols
df_raw = pd.read_csv('indian_pharmaceutical_products_clean.csv')
df_raw = df_raw[df_raw['price_inr'] > 0].copy()
df_raw['brand_name_length'] = df_raw['brand_name'].str.len()
df_raw['price_tier'] = pd.cut(
    df_raw['price_inr'],
    bins=[0, 50, 150, 500, 5000, df_raw['price_inr'].max()],
    labels=['budget', 'low', 'mid', 'premium', 'ultra']
)
df_raw['price_tier_encoded'] = LabelEncoder().fit_transform(df_raw['price_tier'].astype(str))

X_raw = df_raw[raw_feature_cols]
y_raw = df_raw['price_tier_encoded']

X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# Fit the pipeline
full_pipeline.fit(X_tr, y_tr)
y_pred = full_pipeline.predict(X_te)

print("Full Pipeline Training Complete ✅")
print(f"\nClassification Report (Price Tier Prediction):")
print(classification_report(y_te, y_pred,
      target_names=['budget','low','mid','premium','ultra']))


# ============================================================
# STEP 15: SAVE OUTPUTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 15: SAVE OUTPUTS")
print("=" * 60)

# Save main preprocessed dataset
df.to_csv('pharma_preprocessed.csv', index=False)
print(f"✅ pharma_preprocessed.csv  — shape: {df.shape}")

# Save train/val/test splits
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv',   index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv',   index=False)
y_test.to_csv('y_test.csv', index=False)
print("✅ Train / Val / Test splits saved")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print(f"Original shape        : (253973, 15)")
print(f"Final shape           : {df.shape}")
print(f"Missing values        : {df.isnull().sum().sum()}")
print(f"Duplicates removed    : 0")
print(f"Invalid prices removed: 4")
print(f"New features added    : 8")
print(f"Columns encoded       : dosage_form, pack_unit, therapeutic_class, is_discontinued")
print(f"Scalers applied       : StandardScaler, MinMaxScaler")
print(f"Train size            : {X_train.shape[0]:,}")
print(f"Val size              : {X_val.shape[0]:,}")
print(f"Test size             : {X_test.shape[0]:,}")
print("=" * 60)
print("ALL DONE ✅")

# ============================================================
# Part 1 - CareLoop Clinics - No-Show Guard
# 1.1 se 1.6 tak - FINAL CORRECT CODE
# ============================================================
import csv
import json
import pandas as pd
import numpy as np
import requests

# ---------- 1.1 : Read with csv module ONLY ----------
print("="*50)
print("1.1 - Read with csv module")
print("="*50)

with open('data/appointments.csv', newline='') as f:
    reader = csv.DictReader(f)
    print("Header:", reader.fieldnames)
    rows = list(reader)

print("Total rows read:", len(rows))  # 1020

def parse_row(row):
    new_row = dict(row)
    # SIRF type conversion - koi validity check NAHI
    # blank age / blank distance pe hi ValueError aayega
    new_row['age'] = int(new_row['age'])
    new_row['fee_inr'] = int(new_row['fee_inr'])
    new_row['prior_no_shows'] = int(new_row['prior_no_shows'])
    new_row['distance_km'] = float(new_row['distance_km'])
    return new_row

accepted = []
rejected = []
for row in rows:
    try:
        accepted.append(parse_row(row))
    except ValueError:
        rejected.append(row)

print(f"Total read: {len(rows)}, Accepted: {len(accepted)}, Rejected: {len(rejected)}")
# Expected: 1020, 988, 32

with open('data/rejected_rows.json', 'w') as f:
    json.dump(rejected, f, indent=2)
print("Saved rejected_rows.json")

# ---------- 1.2 : Inspect and clean, pass 1 ----------
print("\n" + "="*50)
print("1.2 - Inspect and clean pass 1")
print("="*50)

# IMPORTANT: 1.1 ka accepted use NAHI karna, ORIGINAL se start
df = pd.read_csv('data/appointments.csv')
print("Shape before:", df.shape)  # (1020, 14)
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicated rows:", df.duplicated().sum())  # 20

# (a) Pehle duplicates drop
df = df.drop_duplicates()
print("After dropping duplicates:", df.shape)  # 1000

# (b) booking_channel clean
print("booking_channel before:", df['booking_channel'].nunique())  # 16
df['booking_channel'] = df['booking_channel'].str.strip().str.title()
print("booking_channel after:", df['booking_channel'].unique())  # 3

# (c) Dates convert
df['appt_date'] = pd.to_datetime(df['appt_date'], errors='coerce')
df['booked_on'] = pd.to_datetime(df['booked_on'])

# (d) Fir missing appt_date / status drop
df = df.dropna(subset=['appt_date', 'status'])
print("After dropping missing date/status:", df.shape)  # 990

# ---------- 1.3 : Clean pass 2 ----------
print("\n" + "="*50)
print("1.3 - Validity, gaps, outliers")
print("="*50)

# (a) age <1 ya >100 -> missing
invalid = ((df['age'] < 1) | (df['age'] > 100)).sum()
print(f"Invalid ages: {invalid}")  # 5
df.loc[(df['age'] < 1) | (df['age'] > 100), 'age'] = np.nan

# (b) Median fill
print("Missing age:", df['age'].isnull().sum())  # 17 (12 blank + 5 invalid)
print("Missing distance:", df['distance_km'].isnull().sum())  # 20
age_median = df['age'].median()
dist_median = df['distance_km'].median()
print(f"Median age used: {age_median}, Median distance used: {dist_median}")  # 48.0, 8.25

df['age'] = df['age'].fillna(age_median).round().astype(int)
df['distance_km'] = df['distance_km'].fillna(dist_median)

# (c) Cap SIRF distance_km aur fee_inr ko - age ko NAHI
for col in ['distance_km', 'fee_inr']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    capped = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = np.clip(df[col], lower, upper)
    print(f"1.3 - {col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, Lower={lower:.2f}, Upper={upper:.2f}, Capped={capped}")

# (d) Rounding
df['distance_km'] = df['distance_km'].round(1)
df['fee_inr'] = df['fee_inr'].round().astype(int)
print("After 1.3 shape:", df.shape)

# ---------- 1.4 : Features + NumPy summary ----------
print("\n" + "="*50)
print("1.4 - Features and NumPy summary")
print("="*50)

df['lead_days'] = (df['appt_date'] - df['booked_on']).dt.days
print(f"Lead days min {df['lead_days'].min()}, max {df['lead_days'].max()}, mean {df['lead_days'].mean():.2f}")  # 1, 45

no_show_rate = (df['status'] == 'No-Show').mean()
print(f"No-show: {(df['status'] == 'No-Show').sum()} / {len(df)} = {no_show_rate:.4f}")  # 218 / 990 = 22.0%

# NumPy means - 4 columns, NO LOOP
numeric_cols = ['age', 'distance_km', 'fee_inr', 'lead_days']
arr = df[numeric_cols].to_numpy()
print("Column means [age, distance, fee, lead_days]:")
print(arr.mean(axis=0))  # [48.17, 9.50, 635.71, 11.82]

print(f"lead_days > 14: {(df['lead_days'] > 14).sum()} rows = {(df['lead_days'] > 14).mean()*100:.1f}%")  # 296, 29.9%

# |z| >= 2 for distance_km
z = (df['distance_km'] - df['distance_km'].mean()) / df['distance_km'].std()
print(f"|z| >= 2 for distance_km: {(z.abs() >= 2).sum()} rows")  # 53

# ---------- 1.5 : Enrich clinics from live API ----------
print("\n" + "="*50)
print("1.5 - Clinics API")
print("="*50)

clinics = pd.read_csv('data/clinics.csv')
print("Clinics:", clinics.shape)  # (8, 6)
print(clinics)

temps = []
for _, row in clinics.iterrows():
    temp_c = None
    reason = ""
    for attempt in range(2):  # 1 retry
        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": row['latitude'],
                    "longitude": row['longitude'],
                    "hourly": "temperature_2m",
                    "forecast_days": 1
                },
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if "hourly" in data and "temperature_2m" in data["hourly"]:
                    temp_c = data["hourly"]["temperature_2m"][0]
                    break
                else:
                    reason = "hourly missing"
            else:
                reason = f"status {resp.status_code}"
        except requests.exceptions.RequestException as e:
            reason = str(e)

    if temp_c is None:
        print(f"FAILED: {row['clinic_name']} - {reason}")
    else:
        print(f"{row['clinic_id']} - {row['clinic_name']} - Temp: {temp_c}")

    temps.append(temp_c)

clinics['temp_c'] = temps
print("Temperature nulls:", clinics['temp_c'].isnull().sum())
clinics.to_csv('data/clinics_enriched.csv', index=False)
print("Saved data/clinics_enriched.csv")

# ---------- 1.6 : Export clean table + SQL ----------
print("\n" + "="*50)
print("1.6 - Export")
print("="*50)

# no_show column
df['no_show'] = (df['status'] == 'No-Show').astype(int)

# Sort + Date format
df = df.sort_values('appointment_id')
df['booked_on'] = pd.to_datetime(df['booked_on']).dt.strftime('%Y-%m-%d')
df['appt_date'] = pd.to_datetime(df['appt_date']).dt.strftime('%Y-%m-%d')

cols_16 = ['appointment_id','patient_id','clinic_id','booked_on','appt_date',
           'age','gender','distance_km','fee_inr','booking_channel',
           'visit_type','reminder_sent','prior_no_shows','status',
           'lead_days','no_show']

print("Cols:", len(cols_16), "Missing:", df[cols_16].isnull().sum().sum())
assert df['appointment_id'].is_unique
assert df[cols_16].isnull().sum().sum() == 0

df[cols_16].to_csv('data/appointments_clean.csv', index=False)
print(f"Saved data/appointments_clean.csv - Rows: {len(df)}, Cols: 16")

# SQL - pehle clinics, fir appointments
with open('sql/load_data.sql', 'w') as f:
    for _, r in clinics.iterrows():
        temp_val = "NULL" if pd.isna(r['temp_c']) else str(r['temp_c'])
        f.write(f"INSERT INTO clinics VALUES ('{r['clinic_id']}', '{r['clinic_name']}', '{r['city']}', '{r['tier']}', {r['latitude']}, {r['longitude']}, {temp_val});\n")

    for _, r in df.iterrows():
        f.write(
            f"INSERT INTO appointments VALUES ('{r['appointment_id']}', '{r['patient_id']}', "
            f"'{r['clinic_id']}', '{r['booked_on']}', '{r['appt_date']}', {r['age']}, "
            f"'{r['gender']}', {r['distance_km']}, {r['fee_inr']}, '{r['booking_channel']}', "
            f"'{r['visit_type']}', '{r['reminder_sent']}', {r['prior_no_shows']}, "
            f"'{r['status']}', {r['lead_days']}, {r['no_show']});\n"
        )

print("Saved sql/load_data.sql")
print("\nDONE - Part 1 Complete!")
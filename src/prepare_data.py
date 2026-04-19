import pandas as pd
import numpy as np
import torch
import os
from math import radians, cos, sin, asin, sqrt

# ==========================================
# CONFIGURATION
# ==========================================

# File Paths
PATH_AIMS = "data/raw/coral cover/GBR/[RAW] ltmp_hc_sc_a_by_site.csv"
PATH_REEFCLOUD = "data/raw/coral cover/EcoRRAP_Benthic_Data_2021to2024/reefcloudcover42_functionalgroup42.csv"

# PATH_MOOREA = "data/raw/coral cover/Initial Set/moorea_avg_coral_cover.csv"
# PATH_STJOHN = "data/raw/coral cover/Initial Set/st_john_avg_coral_cover.csv"

PATH_ENV = "data/raw/seawater temperature/Environmental_History_1985-2024.csv" 

# Output
OUTPUT_DIR = "data/processed/"

# Graph Settings
# Connection Threshold (in Kilometers): Reefs closer than this will be connected in the graph
GRAPH_THRESHOLD_KM = 50.0

START_DATE = '1992-06-17'
END_DATE = '2024-03-01'

def ingest_biology():
    print("--- STEP 1: INGESTING GLOBAL BIOLOGICAL DATA ---")
    
    # 1. AIMS (GBR)
    try:
        df_aims = pd.read_csv(PATH_AIMS)
        if 'GROUP_CODE' in df_aims.columns:
            df_aims = df_aims[df_aims['GROUP_CODE'].isin(['Hard Coral'])]
        aims_clean = df_aims.rename(columns={'REEF_ID': 'Site_ID', 'SAMPLE_DATE': 'Date', 'COVER': 'Coral_Cover', 'LATITUDE': 'Latitude', 'LONGITUDE': 'Longitude'})[['Site_ID', 'Date', 'Coral_Cover', 'Latitude', 'Longitude']].copy()
        aims_clean['Coral_Cover'] = pd.to_numeric(aims_clean['Coral_Cover'], errors='coerce') / 100.0
        print(f"   > AIMS (GBR): Loaded {len(aims_clean)} surveys.")
    except Exception as e: print(f"Error AIMS: {e}"); aims_clean = pd.DataFrame()

    # 2. ReefCloud (GBR)
    try:
        df_rc = pd.read_csv(PATH_REEFCLOUD)
        rc_clean = df_rc.rename(columns={'reef': 'Site_ID', 'date': 'Date', 'HC': 'Coral_Cover', 'site_latitude': 'Latitude', 'site_longitude': 'Longitude'})[['Site_ID', 'Date', 'Coral_Cover', 'Latitude', 'Longitude']].copy()
        rc_clean['Coral_Cover'] = pd.to_numeric(rc_clean['Coral_Cover'], errors='coerce') / 100.0
        rc_clean['Date'] = pd.to_datetime(rc_clean['Date'], format='%Y%m', errors='coerce').fillna(pd.to_datetime(rc_clean['Date'], errors='coerce'))
        print(f"   > ReefCloud (GBR): Loaded {len(rc_clean)} surveys.")
    except Exception as e: print(f"Error ReefCloud: {e}"); rc_clean = pd.DataFrame()

    # # 3. Moorea (French Polynesia)
    # try:
    #     df_moorea = pd.read_csv(PATH_MOOREA)
        
    #     # Safely map whatever column names exist in the CSV to standard names
    #     moorea_map = {'Site': 'Site_ID', 'Year': 'Date', 'Stony_coral_cover': 'Coral_Cover'}
    #     df_moorea = df_moorea.rename(columns=moorea_map)
        
    #     # Only extract the columns we know exist
    #     moorea_clean = df_moorea[['Site_ID', 'Date', 'Coral_Cover']].copy()
        
    #     # Format Dates (if it's just a 4-digit year like '2005', make it '2005-06-01')
    #     if str(moorea_clean['Date'].iloc[0]).isdigit() and len(str(moorea_clean['Date'].iloc[0])) == 4:
    #         moorea_clean['Date'] = pd.to_datetime(moorea_clean['Date'].astype(str) + '-06-01')
    #     else:
    #         moorea_clean['Date'] = pd.to_datetime(moorea_clean['Date'])
            
    #     # Format Cover (Divide by 100 if it's a percentage)
    #     moorea_clean['Coral_Cover'] = pd.to_numeric(moorea_clean['Coral_Cover'], errors='coerce')
    #     if moorea_clean['Coral_Cover'].max() > 1.0:
    #         moorea_clean['Coral_Cover'] = moorea_clean['Coral_Cover'] / 100.0
            
    #     # Hardcode approximate Moorea GPS coordinates since they aren't in the CSV
    #     moorea_clean['Latitude'] = -17.53
    #     moorea_clean['Longitude'] = -149.83
        
    #     # Fix Moorea Names
    #     def fix_moorea_name(x):
    #         x_str = str(x).strip()
    #         if x_str.isdigit(): return f"Moorea LTER {x_str}"
    #         elif "LTER" in x_str and "Moorea" not in x_str: return x_str.replace("LTER", "Moorea LTER")
    #         return x_str
            
    #     moorea_clean['Site_ID'] = moorea_clean['Site_ID'].apply(fix_moorea_name)
    #     print(f"   > Moorea (LTER): Loaded {len(moorea_clean)} surveys.")
    # except Exception as e: print(f"Error Moorea: {e}"); moorea_clean = pd.DataFrame()

    # # 4. St. John (USVI)
    # try:
    #     df_stjohn = pd.read_csv(PATH_STJOHN)
        
    #     # Safely map St. John column names
    #     stjohn_map = {'Site': 'Site_ID', 'Year': 'Date', 'Stony_coral_cover': 'Coral_Cover'}
    #     df_stjohn = df_stjohn.rename(columns=stjohn_map)
        
    #     # Extract only the existing columns
    #     stjohn_clean = df_stjohn[['Site_ID', 'Date', 'Coral_Cover']].copy()
        
    #     # Format Dates and Cover
    #     stjohn_clean['Date'] = pd.to_datetime(stjohn_clean['Date'].astype(str) + '-06-01')
    #     stjohn_clean['Coral_Cover'] = pd.to_numeric(stjohn_clean['Coral_Cover'], errors='coerce')
    #     if stjohn_clean['Coral_Cover'].max() > 1.0:
    #         stjohn_clean['Coral_Cover'] = stjohn_clean['Coral_Cover'] / 100.0
        
    #     # Hardcode St. John GPS coordinates
    #     stjohn_clean['Latitude'] = 18.315
    #     stjohn_clean['Longitude'] = -64.725
    #     print(f"   > St. John (USVI): Loaded {len(stjohn_clean)} surveys.")
    # except Exception as e: print(f"Error St. John: {e}"); stjohn_clean = pd.DataFrame()

    # Merge All (Isolate GBR for now)
    full_df = pd.concat([aims_clean, rc_clean], ignore_index=True)
    # full_df = pd.concat([aims_clean, rc_clean, moorea_clean, stjohn_clean], ignore_index=True)

    full_df['Date'] = pd.to_datetime(full_df['Date'])

    # --- TIMELINE AUDIT ---
    print("\n--- BIOLOGICAL TIMELINE AUDIT ---")
    print(f"Absolute Earliest Record: {full_df['Date'].min().date()}")
    print(f"Absolute Latest Record:   {full_df['Date'].max().date()}")
    # print("\nObservation Count per Year:")
    # Count how many surveys happened each year and print them in order
    # yearly_counts = full_df['Date'].dt.year.value_counts().sort_index()
    # for year, count in yearly_counts.items():
    #     print(f"  {year}: {count} surveys")
    print("---------------------------------\n")
    
    # 1. Trim Timeline: Drop all data outside audited window
    full_df = full_df[(full_df['Date'] >= START_DATE) & (full_df['Date'] <= END_DATE)]
    
    # 2. Drop sites with < 3 observations
    obs_counts = full_df['Site_ID'].value_counts()
    valid_dense_sites = obs_counts[obs_counts >= 3].index
    full_df = full_df[full_df['Site_ID'].isin(valid_dense_sites)]
    
    full_df = full_df.dropna(subset=['Coral_Cover', 'Date', 'Site_ID'])
    
    final_bio = full_df.groupby(['Site_ID', 'Date']).agg({'Coral_Cover': 'mean', 'Latitude': 'first', 'Longitude': 'first'}).reset_index()
    print(f"   > Total Global Observations: {len(final_bio)}")
    return final_bio

def build_tensors_and_graph(bio_df):
    print("\n--- STEP 2: ALIGNING WITH SATELLITE DATA ---")
    
    bio_df['Date'] = pd.to_datetime(bio_df['Date'], errors='coerce')

    # Load Environmental Data
    env_df = pd.read_csv(PATH_ENV)
    env_df['Date'] = pd.to_datetime(env_df['Date'])
    
    # Trim environmental data to match the exact biological bounds
    env_df = env_df[(env_df['Date'] >= START_DATE) & (env_df['Date'] <= END_DATE)]

    # Duplicate Cabritte Horn temperature data for the other St. John reefs
    # st_john_sites = ['East Tektite', 'Europa Bay', "Neptune's Table", 'West Little Lameshur', 'White Point']
    # cabritte_env = env_df[env_df['Site_ID'] == 'Cabritte Horn']
    
    # if not cabritte_env.empty:
    #     new_env_rows = []
    #     for site in st_john_sites:
    #         temp_df = cabritte_env.copy()
    #         temp_df['Site_ID'] = site
    #         new_env_rows.append(temp_df)
    #     env_df = pd.concat([env_df] + new_env_rows, ignore_index=True)
    
    env_df = env_df.groupby(['Site_ID', 'Date']).mean(numeric_only=True).reset_index()
    
    bio_sites = set(bio_df['Site_ID'].unique())
    env_sites = set(env_df['Site_ID'].unique())
    valid_sites = sorted(list(bio_sites.intersection(env_sites)))

    #Warn bio sites with no env data
    missing_sites = bio_sites - env_sites
    if missing_sites:
        print(f"   ! WARNING: {len(missing_sites)} biological sites have NO environmental data and will be DROPPED:")
        print(f"   ! Missing Sites: {list(missing_sites)[:10]}...") # Shows first 10
        
        # Count how many actual survey rows are being lost
        dropped_surveys = bio_df[bio_df['Site_ID'].isin(missing_sites)]
        print(f"   ! Total biological surveys lost due to missing env data: {len(dropped_surveys)}")
    else:
        print("   > Success: All biological sites matched with environmental data.")

    # Check for observations outside the window
    env_start, env_end = env_df['Date'].min(), env_df['Date'].max()
    out_of_bounds = bio_df[(bio_df['Date'] < env_start) | (bio_df['Date'] > env_end)]
    if not out_of_bounds.empty:
        print(f"   ! WARNING: {len(out_of_bounds)} surveys are outside the environmental date range ({env_start.year}-{env_end.year})")

    print(f"   > Final Connected Sites: {len(valid_sites)}")
    if len(valid_sites) == 0: raise ValueError("No overlapping sites found!")

    bio_df = bio_df[bio_df['Site_ID'].isin(valid_sites)]
    env_df = env_df[env_df['Site_ID'].isin(valid_sites)]
    dates = sorted(env_df['Date'].unique())
    
    # --- Data Trimming ---
    # Calculate the exact date where the 80% train/test split happens
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    print(f"   > Train/Test Split Date: {split_date.strftime('%Y-%m-%d')}")
    
    # Count biological observations that fall strictly in the Training window
    train_bio = bio_df[bio_df['Date'] < split_date]
    train_counts = train_bio['Site_ID'].value_counts()
    
    # Enforce minimum training points and drop sites with insufficient data (Minimum 2 to establish a trend)
    MIN_TRAIN_OBS = 2 
    valid_train_sites = train_counts[train_counts >= MIN_TRAIN_OBS].index.tolist()
    
    dropped_sites = set(valid_sites) - set(valid_train_sites)
    if dropped_sites:
        print(f"   ! Dropped {len(dropped_sites)} sites for having < {MIN_TRAIN_OBS} training observations.")
        
    # Update valid sites and filter dataframes
    valid_sites = sorted(valid_train_sites)
    bio_df = bio_df[bio_df['Site_ID'].isin(valid_sites)]
    env_df = env_df[env_df['Site_ID'].isin(valid_sites)]
    print(f"   > Final Sites Ready for Tensors: {len(valid_sites)}")
    
    print("Constructing Tensors...")
    # Pivot Tables for Speed
    # Pivot Env Data (Rows=Date, Cols=Site)
    # Aligns dates and sites, filling missing combinations with NaN
    # Averages the temperature across multiple transects for the same Reef ID
    # Efficient pivot using reindex to ensure alignment
    sst_pivot = env_df.pivot_table(index='Date', columns='Site_ID', values='SST', aggfunc='mean').reindex(index=dates).ffill()
    dhw_pivot = env_df.pivot_table(index='Date', columns='Site_ID', values='DHW', aggfunc='mean').reindex(index=dates).ffill()
        
    # Reindex to guarantee all valid_sites exist (Satellite data returning NaNs (land mask) are forced to 0.0)
    sst_pivot = sst_pivot.reindex(columns=valid_sites).fillna(0.0)
    dhw_pivot = dhw_pivot.reindex(columns=valid_sites).fillna(0.0)
    
    # Aggregate daily data into raw monthly averages
    sst_monthly_raw = sst_pivot.resample('MS').mean()
    dhw_monthly_raw = dhw_pivot.resample('MS').mean()
    
    # Group by the month of the year (1-12) and subtract the historical mean for that specific month.
    # Erases the normal summer/winter cycle
    sst_anomaly = sst_monthly_raw.groupby(sst_monthly_raw.index.month).transform(lambda x: x - x.mean())
    
    print("Applying Biologically Optimal 3-Month Smoothing...")
    # Apply the 3-month (12-week) window to capture acute bleaching events
    sst_monthly = sst_anomaly.rolling(window=3, min_periods=1).mean().fillna(0.0)
    dhw_monthly = dhw_monthly_raw.rolling(window=3, min_periods=1).mean().fillna(0.0)
    
    monthly_dates = sst_monthly.index
    
    # Initialize Tensors
    # Build X Tensor (Sites, Time, Features)
    # Transpose to (Sites, Time)
    X_sst = torch.tensor(sst_monthly.values.T, dtype=torch.float32).unsqueeze(-1)
    X_dhw = torch.tensor(dhw_monthly.values.T, dtype=torch.float32).unsqueeze(-1)
    X = torch.cat([X_sst, X_dhw], dim=-1) # (Sites, Time, 2)
    
    # Build Y Tensor and Mask; Mapped to Monthly Dates
    Y = torch.zeros((len(valid_sites), len(monthly_dates), 1))
    Mask = torch.zeros((len(valid_sites), len(monthly_dates), 1))
    
    # Helper to map actual dates to weekly index
    date_to_idx = {d: i for i, d in enumerate(monthly_dates)}
    site_to_idx = {s: i for i, s in enumerate(valid_sites)}
    
    for _, row in bio_df.iterrows():
        # Force date into Timestamp object, just in case it's a string
        bio_date = pd.to_datetime(row['Date'])
        
        # Find nearest monthly date(This assumes bio_df['Date'] is datetime)
        closest_date = min(monthly_dates, key=lambda d: abs(d - bio_date))
        
        s_idx = site_to_idx[row['Site_ID']]
        t_idx = date_to_idx[closest_date]
        # Assign
        Y[s_idx, t_idx, 0] = row['Coral_Cover']
        Mask[s_idx, t_idx, 0] = 1.0
    # --- Graph Construction ---
    print("Building Global Graph...")
    # Build the Spatial Graph
    coords = bio_df.groupby('Site_ID')[['Latitude', 'Longitude']].first().reindex(valid_sites)
    lats = coords['Latitude'].values
    lons = coords['Longitude'].values
    num_sites = len(valid_sites)
    
    # Compute Adjacency Matrix
    adj = torch.zeros((num_sites, num_sites))
    
    # Vectorized Haversine (conceptually) or double loop
    for i in range(num_sites):
        for j in range(num_sites):
            if i == j: 
                adj[i,j] = 1 # Self-loop
                continue
            # Using Haversine is safer
            # Haversine formula to calculate the great circle distance between two points on the earth (specified in decimal degrees)
            dlat = radians(lats[j] - lats[i])
            dlon = radians(lons[j] - lons[i])
            a = sin(dlat/2)**2 + cos(radians(lats[i])) * cos(radians(lats[j])) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            dist_km = 6371 * c # Radius of earth in kilometers
            if dist_km < GRAPH_THRESHOLD_KM:
                # Weighted edge/connection: Closer = Stronger signal
                adj[i,j] = 1.0 / (dist_km + 1e-1)  # Avoid div by zero

    return X, Y, Mask, adj, valid_sites, monthly_dates

def main():
    # Execute Pipeline
    # Ingest
    bio_df = ingest_biology()
    if bio_df.empty: return
    # Build
    X, Y, Mask, Adj, sites, dates = build_tensors_and_graph(bio_df)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(X, f"{OUTPUT_DIR}X.pt")
    torch.save(Y, f"{OUTPUT_DIR}y.pt")
    torch.save(Mask, f"{OUTPUT_DIR}mask.pt")
    torch.save(Adj, f"{OUTPUT_DIR}adjacency_matrix.pt")
    pd.DataFrame({'Site_ID': sites}).to_csv(f"{OUTPUT_DIR}site_list.csv", index=False)
    pd.DataFrame({'Date': dates}).to_csv(f"{OUTPUT_DIR}time_dates.csv", index=False)
    print("Success. Tensors generated.")
    print(f"Processed {len(sites)} sites.")
    print(f"X Shape: {X.shape} (Sites, Weeks, Features)")
    print(f"Y Shape: {Y.shape} (Sites, Weeks, 1)")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
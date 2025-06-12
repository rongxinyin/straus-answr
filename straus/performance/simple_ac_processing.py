import pandas as pd
import numpy as np
import os
import glob
import re

def process_ac_data(data_folder="/data/", output_folder="./output/"):
    """
    Simplified function to process AC power data
    """
    
    # 1. Parse all CSV files
    print("Parsing CSV files...")
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    all_data = []
    equipment_summaries = []
    
    for file_path in csv_files:
        try:
            # Extract equipment ID from filename
            filename = os.path.basename(file_path)
            equipment_match = re.search(r'readings_(.+?)_(\d+)_', filename)
            
            if equipment_match:
                equipment_name = equipment_match.group(1).strip()
                equipment_id = equipment_match.group(2)
                meter_id = f"{equipment_name}_{equipment_id}"
            else:
                meter_id = os.path.splitext(filename)[0]
            
            # Read and clean data
            df = pd.read_csv(file_path)
            
            # Standardize column names
            column_mapping = {
                'Time': 'time',
                'Raw mAmp': 'raw_mamp',
                'Ampere': 'ampere', 
                'Volt': 'volt',
                'Power Factor (Total)': 'power_factor',
                'Power (Total)': 'power_total'
            }
            
            df = df.rename(columns=column_mapping)
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            
            # Convert numeric columns
            numeric_cols = ['raw_mamp', 'ampere', 'volt', 'power_factor', 'power_total']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['time']).sort_values('time')
            
            # 2. Calculate metrics for this equipment
            if 'power_total' in df.columns:
                power_data = df['power_total'].dropna()
                
                # Calculate time intervals and energy
                time_diff = df['time'].diff().dt.total_seconds() / 3600
                avg_interval = time_diff.mean()
                total_energy_kwh = (power_data * avg_interval).sum() / 1000
                
                summary = {
                    'meter_id': meter_id,
                    'avg_power_kw': power_data.mean() / 1000,
                    'max_power_kw': power_data.max() / 1000,
                    'total_energy_kwh': total_energy_kwh,
                    'avg_voltage': df['volt'].mean() if 'volt' in df.columns else None,
                    'avg_current': df['ampere'].mean() if 'ampere' in df.columns else None,
                    'avg_power_factor': df['power_factor'].mean() if 'power_factor' in df.columns else None,
                    'duration_hours': (df['time'].max() - df['time'].min()).total_seconds() / 3600,
                    'data_points': len(df)
                }
                equipment_summaries.append(summary)
            
            # 3. Prepare data for unified dataframe
            value_columns = ['raw_mamp', 'ampere', 'volt', 'power_factor', 'power_total']
            existing_cols = [col for col in value_columns if col in df.columns]
            
            if existing_cols:
                melted_df = pd.melt(
                    df, 
                    id_vars=['time'],
                    value_vars=existing_cols,
                    var_name='meter_var',
                    value_name='value'
                )
                melted_df['meter_id'] = meter_id
                all_data.append(melted_df)
            
            print(f"Processed: {meter_id}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # 4. Create unified dataframe
    if all_data:
        unified_df = pd.concat(all_data, ignore_index=True)
        unified_df = unified_df[['time', 'meter_var', 'meter_id', 'value']]
        unified_df = unified_df.sort_values(['time', 'meter_id', 'meter_var']).reset_index(drop=True)
    else:
        unified_df = pd.DataFrame()
    
    # 5. Create summary report
    if equipment_summaries:
        summary_df = pd.DataFrame(equipment_summaries)
        
        print("\n" + "="*60)
        print("AC POWER CONSUMPTION SUMMARY")
        print("="*60)
        print(summary_df.round(2))
        
        total_energy = summary_df['total_energy_kwh'].sum()
        avg_power = summary_df['avg_power_kw'].mean()
        
        print(f"\nTOTALS:")
        print(f"Total Energy: {total_energy:.2f} kWh")
        print(f"Average Power: {avg_power:.2f} kW")
        print(f"Number of Units: {len(summary_df)}")
    
    # 6. Save results
    os.makedirs(output_folder, exist_ok=True)
    
    if not unified_df.empty:
        unified_path = os.path.join(output_folder, "unified_ac_data.csv")
        unified_df.to_csv(unified_path, index=False)
        print(f"\nSaved unified data: {unified_path}")
    
    if equipment_summaries:
        summary_path = os.path.join(output_folder, "equipment_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")
    
    return unified_df, summary_df if equipment_summaries else None

# Quick execution
if __name__ == "__main__":
    unified_data, summary = process_ac_data()
    
    if unified_data is not None and not unified_data.empty:
        print(f"\nUnified dataframe shape: {unified_data.shape}")
        print(f"Columns: {list(unified_data.columns)}")
        print(f"Sample data:")
        print(unified_data.head())

import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def setup_argument_parser():
    """
    Set up command line argument parser for AC data visualization
    """
    parser = argparse.ArgumentParser(
        description='AC Power Data Pre-Process Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Required arguments group
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument(
        '--data-folder', '-d',
        type=str,
        default='./data/',
        help='Path to folder containing CSV files (default: /data/)')
    
    data_group.add_argument(
        '--output-folder', '-o',
        type=str,
        default='./output/',
        help='Path to output folder for results (default: ./output/)')
    
    return parser

class ACPowerDataProcessor:
    """
    A class to process submetered AC power data from CSV files
    """
    
    def __init__(self, data_folder="/data/"):
        self.data_folder = data_folder
        self.raw_dataframes = {}
        self.cleaned_dataframes = {}
        self.unified_dataframe = None
        self.energy_metrics = {}
        
    def parse_all_csv_files(self):
        """
        Parse all CSV files from the data folder
        """
        # Get all CSV files in the data folder
        csv_files = glob.glob(os.path.join(self.data_folder, "*157Z.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_folder}")
            return
        
        print(f"Found {len(csv_files)} CSV files to process...")
        
        for file_path in csv_files:
            try:
                # Extract equipment ID from filename
                filename = os.path.basename(file_path)
                equipment_id = self._extract_equipment_id(filename)
                
                # Read CSV file
                df = pd.read_csv(file_path)
                self.raw_dataframes[equipment_id] = df
                
                print(f"Successfully loaded: {filename} -> Equipment ID: {equipment_id}")
                
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
        
        return self.raw_dataframes
    
    # Parse a spefici csv file as raw data
    def parse_csv_file(self, file_path):
        """
        Parse a single CSV file and extract equipment ID
        """
        try:
            # Extract filename from path
            filename = os.path.basename(file_path)
            equipment_id = self._extract_equipment_id(filename)
            
            # Read CSV file
            df = pd.read_csv(file_path)
            self.raw_dataframes[equipment_id] = df
            
            print(f"Successfully loaded: {filename} -> Equipment ID: {equipment_id}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
        
        return self.raw_dataframes
    
    def _extract_equipment_id(self, filename):
        """
        Extract equipment ID from filename
        Expected format: readings_Equipment Name_ID_timestamp.csv
        """
        # Try to extract equipment name and ID from filename
        # Example: "readings_Roof Top AC 1_20876_20250521T01_31_58.572Z.csv"
        match = re.search(r'readings_(.+?)_(\d+)', filename)
        if match:
            equipment_name = match.group(1).strip('()')
            # print(f"Extracted equipment name: {equipment_name}")
            equipment_id = match.group(2)
            return f"{equipment_name}"
        else:
            # Fallback: use filename without extension
            return os.path.splitext(filename)[0]
    
    def clean_and_standardize_data(self):
        """
        Clean the data and standardize column names for each equipment
        """
        for equipment_id, df in self.raw_dataframes.items():
            try:
                # Create a copy for cleaning
                cleaned_df = df.copy()
                
                # Standardize column names
                column_mapping = {
                    'Time': 'time',
                    'Raw mAmp': 'raw_mamp',
                    'Ampere': 'ampere', 
                    'Volt': 'volt',
                    'Power Factor (Total)': 'power_factor',
                    'Power (Total)': 'power_total'
                }
                
                # Rename columns if they exist
                for old_col, new_col in column_mapping.items():
                    if old_col in cleaned_df.columns:
                        cleaned_df = cleaned_df.rename(columns={old_col: new_col})
                
                # Convert time column to datetime
                if 'time' in cleaned_df.columns:
                    cleaned_df['time'] = pd.to_datetime(cleaned_df['time'], errors='coerce')
                
                # Convert numeric columns
                numeric_cols = ['raw_mamp', 'ampere', 'volt', 'power_factor', 'power_total']
                for col in numeric_cols:
                    if col in cleaned_df.columns:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                # Remove rows with invalid timestamps
                cleaned_df = cleaned_df.dropna(subset=['time'])
                
                # Sort by time
                cleaned_df = cleaned_df.sort_values('time').reset_index(drop=True)
                
                # Add equipment identifier
                cleaned_df['equipment_id'] = equipment_id
                
                self.cleaned_dataframes[equipment_id] = cleaned_df
                
                print(f"Cleaned data for {equipment_id}: {len(cleaned_df)} records")
                
            except Exception as e:
                print(f"Error cleaning data for {equipment_id}: {str(e)}")
        
        return self.cleaned_dataframes
    
    def calculate_energy_metrics(self):
        """
        Calculate energy use metrics for each equipment
        """
        for equipment_id, df in self.cleaned_dataframes.items():
            try:
                metrics = {}
                
                # Basic statistics
                if 'power_total' in df.columns:
                    power_data = df['power_total'].dropna()
                    
                    metrics['avg_power_kw'] = power_data.mean()
                    metrics['max_power_kw'] = power_data.max()
                    metrics['min_power_kw'] = power_data.min()
                    metrics['std_power_kw'] = power_data.std()
                    
                    # Calculate energy consumption (assuming regular intervals)
                    if len(df) > 1:
                        # Calculate time intervals in hours
                        time_diff = df['time'].diff().dt.total_seconds() / 3600
                        avg_interval = time_diff.mean()
                        
                        # Energy = Power × Time
                        total_energy_kwh = (power_data * avg_interval).sum()
                        metrics['total_energy_kwh'] = total_energy_kwh
                        metrics['avg_interval_hours'] = avg_interval
                
                # Voltage statistics
                if 'volt' in df.columns:
                    volt_data = df['volt'].dropna()
                    metrics['avg_voltage'] = volt_data.mean()
                    metrics['voltage_stability'] = volt_data.std()
                
                # Current statistics  
                if 'ampere' in df.columns:
                    current_data = df['ampere'].dropna()
                    metrics['avg_current'] = current_data.mean()
                    metrics['max_current'] = current_data.max()
                
                # Power factor statistics
                if 'power_factor' in df.columns:
                    pf_data = df['power_factor'].dropna()
                    metrics['avg_power_factor'] = pf_data.mean()
                    metrics['min_power_factor'] = pf_data.min()
                
                # Time range
                metrics['start_time'] = df['time'].min()
                metrics['end_time'] = df['time'].max()
                metrics['duration_hours'] = (metrics['end_time'] - metrics['start_time']).total_seconds() / 3600
                metrics['data_points'] = len(df)
                
                self.energy_metrics[equipment_id] = metrics
                
            except Exception as e:
                print(f"Error calculating metrics for {equipment_id}: {str(e)}")
        
        return self.energy_metrics
    
    def create_unified_dataframe(self):
        """
        Merge all data into a complete dataframe with columns: time, meter_var, meter_id, value
        """
        unified_data = []
        
        for equipment_id, df in self.cleaned_dataframes.items():
            # Melt the dataframe to convert columns to rows
            value_columns = ['raw_mamp', 'ampere', 'volt', 'power_factor', 'power_total']
            
            # Select only existing columns
            existing_value_cols = [col for col in value_columns if col in df.columns]
            
            if existing_value_cols and 'time' in df.columns:
                melted_df = pd.melt(
                    df, 
                    id_vars=['time', 'equipment_id'],
                    value_vars=existing_value_cols,
                    var_name='meter_var',
                    value_name='value'
                )
                
                # Rename equipment_id to meter_id
                melted_df = melted_df.rename(columns={'equipment_id': 'meter_id'})
                
                unified_data.append(melted_df)
        
        if unified_data:
            self.unified_dataframe = pd.concat(unified_data, ignore_index=True)
            
            # Sort by time and meter_id
            self.unified_dataframe = self.unified_dataframe.sort_values(['time', 'meter_id', 'meter_var']).reset_index(drop=True)
            
            print(f"Created unified dataframe with {len(self.unified_dataframe)} records")
        
        return self.unified_dataframe
    
    def generate_summary_report(self):
        """
        Generate a summary report of energy use metrics
        """
        if not self.energy_metrics:
            print("No metrics calculated. Run calculate_energy_metrics() first.")
            return
        
        print("\n" + "="*80)
        print("AC POWER CONSUMPTION SUMMARY REPORT")
        print("="*80)
        
        # Create summary dataframe
        summary_data = []
        for equipment_id, metrics in self.energy_metrics.items():
            summary_data.append({
                'Equipment': equipment_id,
                'Avg Power (kW)': metrics.get('avg_power_kw', 0),
                'Max Power (kW)': metrics.get('max_power_kw', 0),
                'Total Energy (kWh)': metrics.get('total_energy_kwh', 0),
                'Avg Voltage (V)': metrics.get('avg_voltage', 0),
                'Avg Current (A)': metrics.get('avg_current', 0),
                'Avg Power Factor': metrics.get('avg_power_factor', 0),
                'Duration (hrs)': metrics.get('duration_hours', 0),
                'Data Points': metrics.get('data_points', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\nSUMMARY TABLE:")
        print("-" * 80)
        print(summary_df.to_string(index=False, float_format='%.2f'))
        
        # Calculate totals
        total_energy = summary_df['Total Energy (kWh)'].sum()
        avg_power = summary_df['Avg Power (kW)'].mean()
        
        print(f"\nOVERALL METRICS:")
        print("-" * 30)
        print(f"Total Energy Consumption: {total_energy:.2f} kWh")
        print(f"Average Power Across All Units: {avg_power:.2f} kW")
        print(f"Number of AC Units: {len(summary_df)}")
        
        # Detailed metrics for each equipment
        print(f"\nDETAILED METRICS BY EQUIPMENT:")
        print("-" * 50)
        
        for equipment_id, metrics in self.energy_metrics.items():
            print(f"\n{equipment_id}:")
            print(f"  Time Range: {metrics.get('start_time', 'N/A')} to {metrics.get('end_time', 'N/A')}")
            print(f"  Average Power: {metrics.get('avg_power_kw', 0):.2f} kW")
            print(f"  Peak Power: {metrics.get('max_power_kw', 0):.2f} kW")
            print(f"  Total Energy: {metrics.get('total_energy_kwh', 0):.2f} kWh")
            print(f"  Power Factor: {metrics.get('avg_power_factor', 0):.3f}")
            print(f"  Voltage Stability (std): {metrics.get('voltage_stability', 0):.2f} V")
        
        return summary_df
    
    def save_results(self, output_folder="./output/"):
        """
        Save all results to files
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Save unified dataframe
        if self.unified_dataframe is not None:
            unified_path = os.path.join(output_folder, "unified_ac_data.csv")
            self.unified_dataframe.to_csv(unified_path, index=False)
            print(f"Saved unified dataframe to {unified_path}")
        
        # Save individual cleaned dataframes
        for equipment_id, df in self.cleaned_dataframes.items():
            filename = f"cleaned_{equipment_id.replace(' ', '_')}.csv"
            filepath = os.path.join(output_folder, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved cleaned data for {equipment_id} to {filepath}")
        
        # Save energy metrics
        if self.energy_metrics:
            metrics_df = pd.DataFrame(self.energy_metrics).T
            metrics_path = os.path.join(output_folder, "energy_metrics.csv")
            metrics_df.to_csv(metrics_path)
            print(f"Saved energy metrics to {metrics_path}")


# Main execution script
def main():
    """
    Main function to run the complete AC power data processing pipeline
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set data folder from command line argument
    data_folder = args.data_folder
    output_folder = args.output_folder
    
    # Initialize processor
    processor = ACPowerDataProcessor(data_folder=data_folder)
    
    # Step 1: Parse all CSV files
    print("Step 1: Parsing CSV files...")
    processor.parse_all_csv_files()
    # processor.parse_csv_file(os.path.join(data_folder, "zira/readings_(Roof Top AC 1_20876)_2025-07-03T04_14_06.775Z.csv"))
    
    # Step 2: Clean and standardize data
    print("\nStep 2: Cleaning and standardizing data...")
    processor.clean_and_standardize_data()
    
    # Step 3: Calculate energy metrics
    print("\nStep 3: Calculating energy metrics...")
    processor.calculate_energy_metrics()
    
    # Step 4: Create unified dataframe
    print("\nStep 4: Creating unified dataframe...")
    processor.create_unified_dataframe()
    
    # Step 5: Generate summary report
    print("\nStep 5: Generating summary report...")
    processor.generate_summary_report()
    
    # Step 6: Save results
    print("\nStep 6: Saving results...")
    processor.save_results(output_folder=output_folder)

    
    print("\nProcessing complete!")
    
    return processor

# Example usage
if __name__ == "__main__":
    # Run the main processing pipeline
    processor = main()
    
    # Access results
    print(f"\nResults available:")
    print(f"- Raw dataframes: {len(processor.raw_dataframes)} files")
    print(f"- Cleaned dataframes: {len(processor.cleaned_dataframes)} files") 
    print(f"- Unified dataframe shape: {processor.unified_dataframe.shape if processor.unified_dataframe is not None else 'None'}")
    print(f"- Energy metrics calculated for: {len(processor.energy_metrics)} equipment units")

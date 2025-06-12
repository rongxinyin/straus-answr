"""
AC Power Data Visualization and Analysis Tool

This script provides comprehensive visualization and analysis capabilities for AC power
consumption data from submetered CSV files. It includes data processing, energy metrics
calculation, and professional-quality figure generation.

COMMAND LINE USAGE:
    # Basic usage
    python ac_visualization.py
    
    # Custom folders and format
    python ac_visualization.py --data-folder /path/to/data --output-folder ./results --format pdf
    
    # Publication-ready figures
    python ac_visualization.py --style white --context paper --format pdf --dpi 300 --no-show
    
    # Presentation figures
    python ac_visualization.py --style darkgrid --context talk --palette bright --font-scale 1.3
    
    # Show all styling options
    python ac_visualization.py --list-styles

PYTHON USAGE:
    from ac_visualization import complete_ac_analysis
    
    unified_df, summary_df = complete_ac_analysis(
        data_folder="/data/",
        output_folder="./output/",
        style='white', 
        context='paper'
    )

REQUIREMENTS:
    - pandas
    - matplotlib  
    - seaborn
    - numpy

Author: AC Power Analysis Tool
Version: 2.0 with argparse support
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
import argparse
import sys
import os
warnings.filterwarnings('ignore')

def setup_argument_parser():
    """
    Set up command line argument parser for AC data visualization
    """
    parser = argparse.ArgumentParser(
        description='AC Power Data Visualization and Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python ac_visualization.py
  
  # Custom data and output folders
  python ac_visualization.py --data-folder /path/to/data --output-folder ./results
  
  # Publication-ready figures
  python ac_visualization.py --style white --context paper --format pdf --dpi 300
  
  # Presentation figures with large fonts
  python ac_visualization.py --style darkgrid --context talk --palette bright --font-scale 1.3
  
  # Batch processing without displaying plots
  python ac_visualization.py --no-show --format pdf --dpi 150
  
  # High-quality poster figures
  python ac_visualization.py --style whitegrid --context poster --format svg --dpi 600 --font-scale 1.5

Available options:
  Styles: darkgrid, whitegrid, dark, white, ticks
  Contexts: paper, notebook, talk, poster  
  Palettes: husl, deep, muted, bright, pastel, dark, colorblind, Set1, Set2, Set3
  Formats: png, jpg, jpeg, pdf, svg, eps, tiff, ps
        """)
    
    # Required arguments group
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument(
        '--data-folder', '-d',
        type=str,
        default='/data/',
        help='Path to folder containing CSV files (default: /data/)')
    
    data_group.add_argument(
        '--output-folder', '-o',
        type=str,
        default='./output/',
        help='Path to output folder for results (default: ./output/)')
    
    # Figure format and quality options
    format_group = parser.add_argument_group('Figure Format Options')
    format_group.add_argument(
        '--format', '-f',
        type=str,
        choices=['png', 'jpg', 'jpeg', 'pdf', 'svg', 'eps', 'tiff', 'ps'],
        default='png',
        help='Figure format (default: png)')
    
    format_group.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure resolution in DPI (default: 300)')
    
    # Seaborn styling options
    style_group = parser.add_argument_group('Seaborn Styling Options')
    style_group.add_argument(
        '--style', '-s',
        type=str,
        choices=['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'],
        default='whitegrid',
        help='Seaborn style (default: whitegrid)')
    
    style_group.add_argument(
        '--context', '-c',
        type=str,
        choices=['paper', 'notebook', 'talk', 'poster'],
        default='notebook',
        help='Seaborn context for font sizes (default: notebook)')
    
    style_group.add_argument(
        '--palette', '-p',
        type=str,
        default='husl',
        help='Seaborn color palette (default: husl)')
    
    style_group.add_argument(
        '--font-scale',
        type=float,
        default=1.0,
        help='Font scaling factor (default: 1.0)')
    
    # Display options
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument(
        '--no-show',
        action='store_true',
        help='Save figures without displaying them')
    
    display_group.add_argument(
        '--existing-data',
        type=str,
        help='Path to existing unified CSV file to visualize (skips data processing)')
    
    display_group.add_argument(
        '--existing-summary',
        type=str,
        help='Path to existing summary CSV file (used with --existing-data)')
    
    # Processing options
    process_group = parser.add_argument_group('Processing Options')
    process_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output')
    
    process_group.add_argument(
        '--list-styles',
        action='store_true',
        help='Show available styling options and exit')
    
    return parser

def show_seaborn_styles():
    """
    Display available seaborn styles and contexts with examples
    """
    print("SEABORN STYLE OPTIONS:")
    print("=" * 50)
    
    styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
    contexts = ['paper', 'notebook', 'talk', 'poster']
    palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind', 'husl', 'Set1', 'Set2']
    
    print("Available Styles:")
    for style in styles:
        print(f"  • {style}")
    
    print("\nAvailable Contexts:")
    for context in contexts:
        print(f"  • {context}")
    
    print("\nPopular Color Palettes:")
    for palette in palettes:
        print(f"  • {palette}")
    
    print("\nRecommended Combinations:")
    print("  📄 Publications: style='white', context='paper', palette='colorblind'")
    print("  📊 Presentations: style='darkgrid', context='talk', palette='bright'")
    print("  📋 Reports: style='whitegrid', context='notebook', palette='muted'")
    print("  🖼️  Posters: style='white', context='poster', palette='deep'")
    print("  💻 Notebooks: style='darkgrid', context='notebook', palette='husl'")

def create_ac_visualizations(unified_df, summary_df, output_folder="./output/", fig_format='png', dpi=300):
    """
    Create comprehensive visualizations for AC power data analysis
    
    Parameters:
    -----------
    unified_df : pd.DataFrame
        Unified dataframe with time, meter_var, meter_id, value columns
    summary_df : pd.DataFrame
        Summary dataframe with equipment metrics
    output_folder : str
        Base output folder path (default: "./output/")
    fig_format : str
        Figure format: 'png', 'jpg', 'pdf', 'svg', 'eps' (default: 'png')
    dpi : int
        Figure resolution in dots per inch (default: 300)
    """
    
    # Validate figure format
    valid_formats = ['png', 'jpg', 'jpeg', 'pdf', 'svg', 'eps', 'tiff', 'ps']
    if fig_format.lower() not in valid_formats:
        print(f"Warning: '{fig_format}' is not a standard format. Using 'png' instead.")
        print(f"Valid formats: {', '.join(valid_formats)}")
        fig_format = 'png'
    
    # Validate seaborn parameters
    valid_styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
    valid_contexts = ['paper', 'notebook', 'talk', 'poster']
    
    if style not in valid_styles:
        print(f"Warning: '{style}' is not a valid style. Using 'whitegrid' instead.")
        print(f"Valid styles: {', '.join(valid_styles)}")
        style = 'whitegrid'
    
    if context not in valid_contexts:
        print(f"Warning: '{context}' is not a valid context. Using 'notebook' instead.")
        print(f"Valid contexts: {', '.join(valid_contexts)}")
        context = 'notebook'
    
    # Set up the plotting style with seaborn
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    sns.set_palette(palette)
    
    # Additional matplotlib settings for better appearance
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    
    # Create figure directory
    import os
    fig_folder = os.path.join(output_folder, "figures")
    os.makedirs(fig_folder, exist_ok=True)
    
    # 1. Power consumption over time by equipment
    plt.figure(figsize=(15, 8))
    
    power_data = unified_df[unified_df['meter_var'] == 'power_total'].copy()
    power_data['value_kw'] = power_data['value'] / 1000  # Convert to kW
    
    # Use seaborn lineplot for better styling
    sns.lineplot(data=power_data, x='time', y='value_kw', hue='meter_id', 
                linewidth=2.5, alpha=0.8)
    
    plt.title('AC Power Consumption Over Time', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Power (kW)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder, f'power_consumption_timeline.{fig_format}'), 
                dpi=dpi, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 2. Energy consumption comparison (bar chart)
    plt.figure(figsize=(12, 6))
    
    if summary_df is not None:
        # Use seaborn barplot for better styling
        ax = sns.barplot(data=summary_df, x='meter_id', y='total_energy_kwh', 
                        palette=palette, alpha=0.8)
        
        plt.title('Total Energy Consumption by AC Unit', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('AC Units', fontsize=12)
        plt.ylabel('Energy Consumption (kWh)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_folder, f'energy_consumption_comparison.{fig_format}'), 
                    dpi=dpi, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 3. Power factor analysis
    plt.figure(figsize=(12, 10))
    
    pf_data = unified_df[unified_df['meter_var'] == 'power_factor'].copy()
    
    if not pf_data.empty:
        # Box plot of power factors using seaborn
        plt.subplot(2, 1, 1)
        sns.boxplot(data=pf_data, x='meter_id', y='value', palette=palette)
        plt.title('Power Factor Distribution by AC Unit', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('AC Units', fontsize=12)
        plt.ylabel('Power Factor', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Power factor over time using seaborn lineplot
        plt.subplot(2, 1, 2)
        sns.lineplot(data=pf_data, x='time', y='value', hue='meter_id', 
                    alpha=0.8, linewidth=2)
        
        plt.title('Power Factor Over Time', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power Factor', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_folder, f'power_factor_analysis.{fig_format}'), 
                    dpi=dpi, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 4. Voltage stability analysis
    plt.figure(figsize=(12, 6))
    
    voltage_data = unified_df[unified_df['meter_var'] == 'volt'].copy()
    
    if not voltage_data.empty:
        sns.lineplot(data=voltage_data, x='time', y='value', hue='meter_id', 
                    alpha=0.8, linewidth=2.5)
        
        plt.title('Voltage Levels Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Voltage (V)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_folder, f'voltage_stability.{fig_format}'), 
                    dpi=dpi, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 5. Current vs Power scatter plot
    plt.figure(figsize=(12, 8))
    
    current_data = unified_df[unified_df['meter_var'] == 'ampere']
    power_data = unified_df[unified_df['meter_var'] == 'power_total']
    
    if not current_data.empty and not power_data.empty:
        # Merge current and power data
        current_power = pd.merge(
            current_data[['time', 'meter_id', 'value']].rename(columns={'value': 'current'}),
            power_data[['time', 'meter_id', 'value']].rename(columns={'value': 'power'}),
            on=['time', 'meter_id']
        )
        
        current_power['power_kw'] = current_power['power'] / 1000  # Convert to kW
        
        # Use seaborn scatterplot for better styling
        sns.scatterplot(data=current_power, x='current', y='power_kw', 
                       hue='meter_id', alpha=0.7, s=60, palette=palette)
        
        plt.title('Current vs Power Relationship', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Current (A)', fontsize=12)
        plt.ylabel('Power (kW)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_folder, f'current_vs_power.{fig_format}'), 
                    dpi=dpi, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 6. Summary dashboard
    if summary_df is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Average power comparison
        sns.barplot(data=summary_df, x='meter_id', y='avg_power_kw', 
                   palette=palette, ax=axes[0, 0])
        axes[0, 0].set_title('Average Power Consumption', fontweight='bold', pad=15)
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].set_xlabel('AC Units')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Power factor comparison
        sns.barplot(data=summary_df, x='meter_id', y='avg_power_factor', 
                   palette=palette, ax=axes[0, 1])
        axes[0, 1].set_title('Average Power Factor', fontweight='bold', pad=15)
        axes[0, 1].set_ylabel('Power Factor')
        axes[0, 1].set_xlabel('AC Units')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Voltage levels
        sns.barplot(data=summary_df, x='meter_id', y='avg_voltage', 
                   palette=palette, ax=axes[1, 0])
        axes[1, 0].set_title('Average Voltage Levels', fontweight='bold', pad=15)
        axes[1, 0].set_ylabel('Voltage (V)')
        axes[1, 0].set_xlabel('AC Units')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Current levels
        sns.barplot(data=summary_df, x='meter_id', y='avg_current', 
                   palette=palette, ax=axes[1, 1])
        axes[1, 1].set_title('Average Current Draw', fontweight='bold', pad=15)
        axes[1, 1].set_ylabel('Current (A)')
        axes[1, 1].set_xlabel('AC Units')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('AC Units Performance Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_folder, f'performance_dashboard.{fig_format}'), 
                    dpi=dpi, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    print(f"All visualizations saved to: {fig_folder}")
    print(f"Settings: {fig_format} format, {dpi} DPI, {style} style, {context} context, {palette} palette")

def generate_detailed_report(unified_df, summary_df, output_folder="./output/"):
    """
    Generate a detailed text report with insights
    """
    
    report_path = os.path.join(output_folder, "detailed_ac_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE AC POWER CONSUMPTION ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if summary_df is not None:
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            total_energy = summary_df['total_energy_kwh'].sum()
            avg_power = summary_df['avg_power_kw'].mean()
            max_power_unit = summary_df.loc[summary_df['avg_power_kw'].idxmax(), 'meter_id']
            min_pf_unit = summary_df.loc[summary_df['avg_power_factor'].idxmin(), 'meter_id']
            
            f.write(f"• Total energy consumed: {total_energy:.2f} kWh\n")
            f.write(f"• Average power consumption: {avg_power:.2f} kW\n")
            f.write(f"• Number of AC units monitored: {len(summary_df)}\n")
            f.write(f"• Highest power consuming unit: {max_power_unit}\n")
            f.write(f"• Unit with lowest power factor: {min_pf_unit}\n\n")
            
            f.write("DETAILED METRICS BY UNIT\n")
            f.write("-" * 40 + "\n")
            
            for _, row in summary_df.iterrows():
                f.write(f"\n{row['meter_id']}:\n")
                f.write(f"  Average Power: {row['avg_power_kw']:.2f} kW\n")
                f.write(f"  Peak Power: {row['max_power_kw']:.2f} kW\n")
                f.write(f"  Total Energy: {row['total_energy_kwh']:.2f} kWh\n")
                f.write(f"  Power Factor: {row['avg_power_factor']:.3f}\n")
                f.write(f"  Average Voltage: {row['avg_voltage']:.1f} V\n")
                f.write(f"  Average Current: {row['avg_current']:.2f} A\n")
                f.write(f"  Monitoring Duration: {row['duration_hours']:.1f} hours\n")
            
            # Performance insights
            f.write("\n\nPERFORMANCE INSIGHTS\n")
            f.write("-" * 40 + "\n")
            
            # Power factor analysis
            poor_pf_units = summary_df[summary_df['avg_power_factor'] < 0.85]
            if not poor_pf_units.empty:
                f.write("⚠️  POWER FACTOR CONCERNS:\n")
                for _, unit in poor_pf_units.iterrows():
                    f.write(f"   - {unit['meter_id']}: PF = {unit['avg_power_factor']:.3f}\n")
                f.write("   Recommendation: Consider power factor correction\n\n")
            
            # Energy efficiency ranking
            f.write("ENERGY EFFICIENCY RANKING:\n")
            efficiency_rank = summary_df.sort_values('avg_power_kw')
            for i, (_, unit) in enumerate(efficiency_rank.iterrows()):
                f.write(f"   {i+1}. {unit['meter_id']}: {unit['avg_power_kw']:.2f} kW\n")
        
        f.write(f"\n\nReport saved to: {report_path}\n")
    
    print(f"Detailed report saved to: {report_path}")

def visualize_existing_data(unified_csv_path, summary_csv_path=None, 
                          output_folder="./output/", fig_format='png', 
                          dpi=300, show_plots=True, style='whitegrid', 
                          context='notebook', palette='husl', font_scale=1.0):
    """
    Create visualizations from existing processed CSV files
    
    Parameters:
    -----------
    unified_csv_path : str
        Path to unified CSV file with columns: time, meter_var, meter_id, value
    summary_csv_path : str, optional
        Path to summary CSV file with equipment metrics
    output_folder : str
        Path to output folder for figures (default: "./output/")
    fig_format : str
        Figure format: 'png', 'jpg', 'pdf', 'svg', 'eps' (default: 'png')
    dpi : int
        Figure resolution in dots per inch (default: 300)
    show_plots : bool
        Whether to display plots interactively (default: True)
    style : str
        Seaborn style: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks' (default: 'whitegrid')
    context : str
        Seaborn context: 'paper', 'notebook', 'talk', 'poster' (default: 'notebook')
    palette : str
        Seaborn color palette (default: 'husl')
    font_scale : float
        Font scaling factor (default: 1.0)
    
    Returns:
    --------
    tuple
        (unified_df, summary_df) - loaded dataframes
    """
    
    # Load unified data
    try:
        unified_df = pd.read_csv(unified_csv_path)
        unified_df['time'] = pd.to_datetime(unified_df['time'])
        print(f"Loaded unified data: {unified_df.shape}")
    except Exception as e:
        print(f"Error loading unified data from {unified_csv_path}: {e}")
        return None, None
    
    # Load summary data if provided
    summary_df = None
    if summary_csv_path:
        try:
            summary_df = pd.read_csv(summary_csv_path)
            print(f"Loaded summary data: {summary_df.shape}")
        except Exception as e:
            print(f"Error loading summary data from {summary_csv_path}: {e}")
    
    # Create visualizations
    create_ac_visualizations(unified_df, summary_df, output_folder,style='whitegrid',)
    
    print("Visualization complete!")
    return unified_df, summary_df

# Example usage function
def complete_ac_analysis(data_folder="/data/", output_folder="./output/", 
                        fig_format='png', dpi=300, show_plots=True,
                        style='whitegrid', context='notebook', palette='husl', 
                        font_scale=1.0):
    """
    Run complete analysis pipeline with visualizations
    
    Parameters:
    -----------
    data_folder : str
        Path to folder containing CSV files (default: "/data/")
    output_folder : str
        Path to output folder for results (default: "./output/")
    fig_format : str
        Figure format: 'png', 'jpg', 'pdf', 'svg', 'eps' (default: 'png')
    dpi : int
        Figure resolution in dots per inch (default: 300)
    show_plots : bool
        Whether to display plots interactively (default: True)
    style : str
        Seaborn style: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks' (default: 'whitegrid')
    context : str
        Seaborn context: 'paper', 'notebook', 'talk', 'poster' (default: 'notebook')
    palette : str
        Seaborn color palette (default: 'husl')
    font_scale : float
        Font scaling factor (default: 1.0)
    
    Returns:
    --------
    tuple
        (unified_df, summary_df) - processed dataframes
    """
    # Import the simple processing function
    from simple_ac_processing import process_ac_data
    
    # Process the data
    unified_df, summary_df = process_ac_data(data_folder, output_folder)
    
    if unified_df is not None and not unified_df.empty:
        # Create visualizations
        create_ac_visualizations(unified_df, summary_df, output_folder, 
                               fig_format, dpi, show_plots, style, 
                               context, palette, font_scale)
        
        # Generate detailed report
        generate_detailed_report(unified_df, summary_df, output_folder)
        
        print("Complete analysis finished!")
        return unified_df, summary_df
    else:
        print("No data to analyze.")
        return None, None

def main():
    """
    Main function for command-line interface
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Show styling options if requested
    if args.list_styles:
        show_seaborn_styles()
        return
    
    # Set up verbose output
    if args.verbose:
        print("AC Power Data Visualization Tool")
        print("=" * 50)
        print(f"Data folder: {args.data_folder}")
        print(f"Output folder: {args.output_folder}")
        print(f"Figure format: {args.format}")
        print(f"DPI: {args.dpi}")
        print(f"Style: {args.style}")
        print(f"Context: {args.context}")
        print(f"Palette: {args.palette}")
        print(f"Font scale: {args.font_scale}")
        print(f"Show plots: {not args.no_show}")
        print("=" * 50)
    
    # Check if data folder exists (unless using existing data)
    if not args.existing_data and not os.path.exists(args.data_folder):
        print(f"Error: Data folder '{args.data_folder}' does not exist.")
        print("Use --data-folder to specify a valid path to your CSV files.")
        sys.exit(1)
    
    try:
        # Option 1: Visualize existing processed data
        if args.existing_data:
            if args.verbose:
                print("Loading existing processed data...")
            
            unified_df, summary_df = visualize_existing_data(
                unified_csv_path=args.existing_data,
                summary_csv_path=args.existing_summary,
                output_folder=args.output_folder,
                fig_format=args.format,
                dpi=args.dpi,
                show_plots=not args.no_show,
                style=args.style,
                context=args.context,
                palette=args.palette,
                font_scale=args.font_scale
            )
            
        # Option 2: Complete analysis pipeline
        else:
            if args.verbose:
                print("Running complete analysis pipeline...")
            
            unified_df, summary_df = complete_ac_analysis(
                data_folder=args.data_folder,
                output_folder=args.output_folder,
                fig_format=args.format,
                dpi=args.dpi,
                show_plots=not args.no_show,
                style=args.style,
                context=args.context,
                palette=args.palette,
                font_scale=args.font_scale
            )
        
        if args.verbose:
            print("\nAnalysis completed successfully!")
            if unified_df is not None:
                print(f"Processed {len(unified_df)} data points")
            if summary_df is not None:
                print(f"Analyzed {len(summary_df)} AC units")
                
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Check if script is being run directly or imported
    if len(sys.argv) > 1:
        # Command line usage
        main()
    else:
        # Interactive usage - show help and examples
        print("AC Power Data Visualization Tool")
        print("=" * 50)
        print("This script can be used in two ways:")
        print()
        print("1. COMMAND LINE (recommended):")
        print("   python ac_visualization.py --help")
        print("   python ac_visualization.py --data-folder /path/to/data")
        print("   python ac_visualization.py --style white --context paper --format pdf")
        print()
        print("2. INTERACTIVE (current mode):")
        print("   Running with default settings...")
        print()
        
        # Show available styling options
        show_seaborn_styles()
        
        # Run with default settings
        print("\nRunning analysis with default settings...")
        try:
            unified_data, summary = complete_ac_analysis()
        except Exception as e:
            print(f"Error: {e}")
            print("\nTry using command line arguments for better control:")
            print("python ac_visualization.py --help")

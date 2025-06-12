#!/usr/bin/env python3
"""
HVAC Power Usage Pattern Analysis
Comprehensive analysis and visualization of power consumption data with configurable styling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import os
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

class HVACPowerAnalyzer:
    def __init__(self, csv_file: str, style: str = 'notebook', output_dir: str = 'plots', 
                 output_format: str = 'png', dpi: int = 300, systems: Optional[List[str]] = None):
        """Initialize the analyzer with CSV data and styling options"""
        self.csv_file = csv_file
        self.style = style
        self.output_dir = output_dir
        self.output_format = output_format
        self.dpi = dpi
        self.systems = systems
        self.data = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup matplotlib style based on selected option
        self._setup_plot_style()
        
        # Load and process data
        self.load_data()
        
    @classmethod
    def get_available_styles(cls) -> Dict[str, str]:
        """Get available plot styles and their descriptions."""
        return {
            'paper': 'Optimized for academic publications',
            'notebook': 'Balanced for interactive analysis',
            'talk': 'Large fonts for presentations',
            'poster': 'Extra large fonts for conference posters'
        }
    
    def get_style_adjusted_params(self) -> Dict:
        """Get style-specific parameters for plots."""
        base_params = {
            'paper': {
                'title_pad': 10,
                'annotation_fontsize': 8,
                'error_capsize': 3,
                'scatter_alpha': 0.6,
                'line_alpha': 0.8,
                'figsize_main': (16, 20),
                'figsize_detail': (14, 10),
                'title_fontsize': 14,
                'label_fontsize': 10,
                'tick_fontsize': 8
            },
            'notebook': {
                'title_pad': 15,
                'annotation_fontsize': 10,
                'error_capsize': 4,
                'scatter_alpha': 0.6,
                'line_alpha': 0.8,
                'figsize_main': (20, 24),
                'figsize_detail': (16, 12),
                'title_fontsize': 16,
                'label_fontsize': 12,
                'tick_fontsize': 10
            },
            'talk': {
                'title_pad': 20,
                'annotation_fontsize': 14,
                'error_capsize': 6,
                'scatter_alpha': 0.7,
                'line_alpha': 0.9,
                'figsize_main': (24, 28),
                'figsize_detail': (20, 16),
                'title_fontsize': 20,
                'label_fontsize': 16,
                'tick_fontsize': 14
            },
            'poster': {
                'title_pad': 25,
                'annotation_fontsize': 18,
                'error_capsize': 8,
                'scatter_alpha': 0.8,
                'line_alpha': 0.9,
                'figsize_main': (28, 32),
                'figsize_detail': (24, 20),
                'title_fontsize': 24,
                'label_fontsize': 20,
                'tick_fontsize': 16
            }
        }
        return base_params[self.style]
    
    def _setup_plot_style(self):
        """Setup matplotlib and seaborn styling based on selected style."""
        # Set base style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Get style parameters
        params = self.get_style_adjusted_params()
        
        # Configure matplotlib rcParams based on style
        plt.rcParams.update({
            'font.size': params['tick_fontsize'],
            'axes.titlesize': params['title_fontsize'],
            'axes.labelsize': params['label_fontsize'],
            'xtick.labelsize': params['tick_fontsize'],
            'ytick.labelsize': params['tick_fontsize'],
            'legend.fontsize': params['tick_fontsize'],
            'figure.titlesize': params['title_fontsize'],
            'lines.linewidth': 2,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        
        logging.info(f"Plot style configured: {self.style}")
        
    def load_data(self):
        """Load and preprocess the CSV data"""
        logging.info("Loading and preprocessing data...")
        
        # Read CSV file
        self.data = pd.read_csv(self.csv_file)
        
        # Convert time column to datetime
        self.data['time'] = pd.to_datetime(self.data['time'])
        
        # Extract time components
        self.data['hour'] = self.data['time'].dt.hour
        self.data['minute'] = self.data['time'].dt.minute
        self.data['date'] = self.data['time'].dt.date
        self.data['time_str'] = self.data['time'].dt.strftime('%H:%M:%S')
        
        # Calculate efficiency metric
        self.data['efficiency'] = self.data['power_factor'] * 100
        
        # Filter by systems if specified
        if self.systems:
            # For the current dataset, we'll use equipment_id filtering
            # In a real HVAC system, you might filter by system type
            logging.info(f"Filtering for systems: {self.systems}")
            # Note: Current data only has 'Roof Top AC 1', so no filtering needed for this example
        
        logging.info(f"Data loaded: {len(self.data)} records")
        logging.info(f"Time range: {self.data['time'].min()} to {self.data['time'].max()}")
        logging.info(f"Equipment: {self.data['equipment_id'].unique()}")
        
    def basic_statistics(self):
        """Calculate and display basic statistics"""
        logging.info("Calculating basic statistics...")
        
        print("\n📊 POWER CONSUMPTION STATISTICS")
        print("=" * 50)
        
        power_stats = self.data['power_total'].describe()
        
        print(f"Total Energy Consumption: {self.data['power_total'].sum():.3f} kWh")
        print(f"Average Power: {power_stats['mean']:.3f} kW")
        print(f"Median Power: {power_stats['50%']:.3f} kW")
        print(f"Peak Power: {power_stats['max']:.3f} kW")
        print(f"Minimum Power: {power_stats['min']:.3f} kW")
        print(f"Standard Deviation: {power_stats['std']:.3f} kW")
        print(f"Power Range: {power_stats['max'] - power_stats['min']:.3f} kW")
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95]
        print(f"\n📈 Power Percentiles:")
        for p in percentiles:
            val = np.percentile(self.data['power_total'], p)
            print(f"  {p}th percentile: {val:.3f} kW")
            
        # Power factor analysis
        pf_stats = self.data['power_factor'].describe()
        print(f"\n⚡ Power Factor Analysis:")
        print(f"  Average: {pf_stats['mean']:.3f}")
        print(f"  Range: {pf_stats['min']:.3f} - {pf_stats['max']:.3f}")
        print(f"  Std Dev: {pf_stats['std']:.3f}")
        
    def hourly_analysis(self):
        """Analyze power consumption patterns by hour"""
        logging.info("Performing hourly analysis...")
        
        print("\n🕐 HOURLY POWER CONSUMPTION PATTERNS")
        print("=" * 50)
        
        hourly_stats = self.data.groupby('hour')['power_total'].agg([
            'count', 'mean', 'min', 'max', 'std'
        ]).round(3)
        
        print("Hour | Count | Avg Power | Min Power | Max Power | Std Dev")
        print("-" * 60)
        for hour, stats in hourly_stats.iterrows():
            print(f"{hour:2d}:00 |  {stats['count']:.0f}  |   {stats['mean']:6.3f}  |   {stats['min']:6.3f}  |   {stats['max']:6.3f}  | {stats['std']:6.3f}")
            
        return hourly_stats
    
    def detect_patterns(self):
        """Detect operational patterns and anomalies"""
        logging.info("Detecting operational patterns...")
        
        print("\n🔍 OPERATIONAL PATTERN DETECTION")
        print("=" * 50)
        
        # Define power thresholds
        q25 = np.percentile(self.data['power_total'], 25)
        q75 = np.percentile(self.data['power_total'], 75)
        
        # Categorize power levels
        standby_mask = self.data['power_total'] <= q25
        active_mask = self.data['power_total'] >= q75
        transition_mask = ~(standby_mask | active_mask)
        
        standby_count = standby_mask.sum()
        active_count = active_mask.sum()
        transition_count = transition_mask.sum()
        
        print(f"🟢 Standby Mode (≤{q25:.3f} kW): {standby_count} readings ({standby_count/len(self.data)*100:.1f}%)")
        print(f"🔴 Active Mode (≥{q75:.3f} kW): {active_count} readings ({active_count/len(self.data)*100:.1f}%)")
        print(f"🟡 Transition Mode: {transition_count} readings ({transition_count/len(self.data)*100:.1f}%)")
        
        # Energy distribution
        standby_energy = self.data[standby_mask]['power_total'].sum()
        active_energy = self.data[active_mask]['power_total'].sum()
        transition_energy = self.data[transition_mask]['power_total'].sum()
        total_energy = self.data['power_total'].sum()
        
        print(f"\n⚡ Energy Distribution:")
        print(f"  Standby Energy: {standby_energy:.3f} kWh ({standby_energy/total_energy*100:.1f}%)")
        print(f"  Active Energy: {active_energy:.3f} kWh ({active_energy/total_energy*100:.1f}%)")
        print(f"  Transition Energy: {transition_energy:.3f} kWh ({transition_energy/total_energy*100:.1f}%)")
        
        return {
            'standby_threshold': q25,
            'active_threshold': q75,
            'standby_count': standby_count,
            'active_count': active_count,
            'transition_count': transition_count
        }
    
    def moving_average(self, window_size=5):
        """Calculate moving average for trend analysis"""
        return self.data['power_total'].rolling(window=window_size, center=True).mean()
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        logging.info("Creating comprehensive visualizations...")
        
        # Get style parameters
        params = self.get_style_adjusted_params()
        
        # Create figure with subplots
        fig = plt.figure(figsize=params['figsize_main'])
        
        # 1. Power Consumption Timeline
        ax1 = plt.subplot(3, 2, 1)
        plt.plot(self.data['time'], self.data['power_total'], 
                linewidth=2, color='#2E86AB', alpha=params['line_alpha'], label='Actual Power')
        
        # Add moving average
        ma = self.moving_average()
        plt.plot(self.data['time'], ma, 
                linewidth=3, color='#F24236', alpha=params['line_alpha'], label='5-Point Moving Avg')
        
        plt.title('📈 Power Consumption Timeline', fontsize=params['title_fontsize'], 
                 fontweight='bold', pad=params['title_pad'])
        plt.xlabel('Time', fontsize=params['label_fontsize'])
        plt.ylabel('Power (kW)', fontsize=params['label_fontsize'])
        plt.legend(fontsize=params['tick_fontsize'])
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. Power Distribution Histogram
        ax2 = plt.subplot(3, 2, 2)
        plt.hist(self.data['power_total'], bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
        plt.axvline(self.data['power_total'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.data["power_total"].mean():.3f} kW')
        plt.axvline(self.data['power_total'].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {self.data["power_total"].median():.3f} kW')
        plt.title('📊 Power Distribution', fontsize=params['title_fontsize'], 
                 fontweight='bold', pad=params['title_pad'])
        plt.xlabel('Power (kW)', fontsize=params['label_fontsize'])
        plt.ylabel('Frequency', fontsize=params['label_fontsize'])
        plt.legend(fontsize=params['tick_fontsize'])
        plt.grid(True, alpha=0.3)
        
        # 3. Hourly Power Patterns
        ax3 = plt.subplot(3, 2, 3)
        hourly_stats = self.data.groupby('hour')['power_total'].agg(['mean', 'min', 'max'])
        
        hours = hourly_stats.index
        plt.bar(hours, hourly_stats['mean'], color='#F18F01', alpha=0.7, label='Average')
        plt.errorbar(hours, hourly_stats['mean'], 
                    yerr=[hourly_stats['mean'] - hourly_stats['min'], 
                          hourly_stats['max'] - hourly_stats['mean']], 
                    fmt='none', color='black', alpha=0.5, capsize=params['error_capsize'])
        
        plt.title('🕐 Hourly Power Patterns', fontsize=params['title_fontsize'], 
                 fontweight='bold', pad=params['title_pad'])
        plt.xlabel('Hour of Day', fontsize=params['label_fontsize'])
        plt.ylabel('Power (kW)', fontsize=params['label_fontsize'])
        plt.legend(fontsize=params['tick_fontsize'])
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        
        # 4. Power vs Efficiency Scatter Plot
        ax4 = plt.subplot(3, 2, 4)
        scatter = plt.scatter(self.data['power_total'], self.data['power_factor'], 
                            c=self.data['hour'], cmap='viridis', alpha=params['scatter_alpha'], s=50)
        plt.colorbar(scatter, label='Hour of Day')
        plt.title('⚡ Power vs Power Factor', fontsize=params['title_fontsize'], 
                 fontweight='bold', pad=params['title_pad'])
        plt.xlabel('Power (kW)', fontsize=params['label_fontsize'])
        plt.ylabel('Power Factor', fontsize=params['label_fontsize'])
        plt.grid(True, alpha=0.3)
        
        # 5. Box Plot by Hour
        ax5 = plt.subplot(3, 2, 5)
        # Filter hours with data
        hours_with_data = self.data.groupby('hour')['power_total'].count()
        hours_to_plot = hours_with_data[hours_with_data > 0].index
        
        box_data = [self.data[self.data['hour'] == h]['power_total'] for h in hours_to_plot]
        plt.boxplot(box_data, labels=hours_to_plot)
        plt.title('📦 Power Distribution by Hour', fontsize=params['title_fontsize'], 
                 fontweight='bold', pad=params['title_pad'])
        plt.xlabel('Hour of Day', fontsize=params['label_fontsize'])
        plt.ylabel('Power (kW)', fontsize=params['label_fontsize'])
        plt.grid(True, alpha=0.3)
        
        # 6. Cumulative Energy Consumption
        ax6 = plt.subplot(3, 2, 6)
        cumulative_energy = self.data['power_total'].cumsum()
        plt.plot(self.data['time'], cumulative_energy, linewidth=3, color='#C73E1D')
        plt.title('📈 Cumulative Energy Consumption', fontsize=params['title_fontsize'], 
                 fontweight='bold', pad=params['title_pad'])
        plt.xlabel('Time', fontsize=params['label_fontsize'])
        plt.ylabel('Cumulative Energy (kWh)', fontsize=params['label_fontsize'])
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout(pad=3.0)
        
        # Save with appropriate filename and format
        filename = os.path.join(self.output_dir, f'hvac_power_analysis_{self.style}.{self.output_format}')
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        logging.info(f"Main analysis plot saved: {filename}")
        # plt.show()
        
    def create_detailed_analysis_plots(self):
        """Create additional detailed analysis plots"""
        logging.info("Creating detailed analysis plots...")
        
        # Get style parameters
        params = self.get_style_adjusted_params()
        
        # Create second figure for detailed analysis
        fig, axes = plt.subplots(2, 2, figsize=params['figsize_detail'])
        
        # 1. Power Pattern Heatmap
        ax1 = axes[0, 0]
        # Create hour vs time matrix
        self.data['time_minutes'] = self.data['time'].dt.hour * 60 + self.data['time'].dt.minute
        pivot_data = self.data.pivot_table(values='power_total', 
                                          index='hour', 
                                          columns='minute', 
                                          aggfunc='mean')
        
        sns.heatmap(pivot_data, ax=ax1, cmap='YlOrRd', cbar_kws={'label': 'Power (kW)'})
        ax1.set_title('🔥 Power Consumption Heatmap (Hour vs Minute)', 
                     fontsize=params['title_fontsize'], fontweight='bold', pad=params['title_pad'])
        ax1.set_xlabel('Minute', fontsize=params['label_fontsize'])
        ax1.set_ylabel('Hour', fontsize=params['label_fontsize'])
        
        # 2. Power Trend with Annotations
        ax2 = axes[0, 1]
        ax2.plot(self.data['time'], self.data['power_total'], linewidth=2, color='#2E86AB')
        
        # Annotate key points
        max_power_idx = self.data['power_total'].idxmax()
        ax2.annotate(f'Peak: {self.data.loc[max_power_idx, "power_total"]:.3f} kW', 
                    xy=(self.data.loc[max_power_idx, 'time'], self.data.loc[max_power_idx, 'power_total']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    fontsize=params['annotation_fontsize'])
        
        ax2.set_title('📈 Power Trend with Key Points', fontsize=params['title_fontsize'], 
                     fontweight='bold', pad=params['title_pad'])
        ax2.set_xlabel('Time', fontsize=params['label_fontsize'])
        ax2.set_ylabel('Power (kW)', fontsize=params['label_fontsize'])
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Power Factor vs Power Correlation
        ax3 = axes[1, 0]
        ax3.scatter(self.data['power_total'], self.data['power_factor'], 
                   alpha=params['scatter_alpha'])
        
        # Add correlation coefficient
        correlation = np.corrcoef(self.data['power_total'], self.data['power_factor'])[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax3.transAxes, fontsize=params['annotation_fontsize'],
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax3.set_title('⚡ Power vs Power Factor Correlation', fontsize=params['title_fontsize'], 
                     fontweight='bold', pad=params['title_pad'])
        ax3.set_xlabel('Power (kW)', fontsize=params['label_fontsize'])
        ax3.set_ylabel('Power Factor', fontsize=params['label_fontsize'])
        ax3.grid(True, alpha=0.3)
        
        # 4. Operating Mode Analysis
        ax4 = axes[1, 1]
        patterns = self.detect_patterns()
        
        # Create pie chart of operating modes
        sizes = [patterns['standby_count'], patterns['active_count'], patterns['transition_count']]
        labels = ['Standby Mode', 'Active Mode', 'Transition Mode']
        colors = ['#3498db', '#e74c3c', '#f39c12']
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': params['tick_fontsize']})
        ax4.set_title('🔄 Operating Mode Distribution', fontsize=params['title_fontsize'], 
                     fontweight='bold', pad=params['title_pad'])
        
        plt.tight_layout()
        
        # Save with appropriate filename and format
        filename = os.path.join(self.output_dir, f'hvac_detailed_analysis_{self.style}.{self.output_format}')
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        logging.info(f"Detailed analysis plot saved: {filename}")
        # plt.show()
        
    def generate_insights(self):
        """Generate key insights and recommendations"""
        logging.info("Generating insights and recommendations...")
        
        print("\n🔍 KEY INSIGHTS & RECOMMENDATIONS")
        print("=" * 60)
        
        # Calculate key metrics
        total_energy = self.data['power_total'].sum()
        avg_power = self.data['power_total'].mean()
        max_power = self.data['power_total'].max()
        min_power = self.data['power_total'].min()
        
        # Time analysis
        duration_hours = (self.data['time'].max() - self.data['time'].min()).total_seconds() / 3600
        
        # Operating efficiency
        avg_power_factor = self.data['power_factor'].mean()
        
        # Peak hours analysis
        high_power_mask = self.data['power_total'] > avg_power * 2
        peak_hours = self.data[high_power_mask]['hour'].unique()
        
        print(f"🏢 EQUIPMENT: {self.data['equipment_id'].iloc[0]}")
        print(f"📅 MONITORING PERIOD: {duration_hours:.1f} hours")
        print(f"⚡ TOTAL ENERGY: {total_energy:.3f} kWh")
        print(f"📊 POWER VARIATION: {min_power:.3f} - {max_power:.3f} kW ({max_power/min_power:.1f}x range)")
        print(f"🎯 AVERAGE EFFICIENCY: {avg_power_factor*100:.1f}%")
        
        print(f"\n🔥 PEAK OPERATION HOURS: {sorted(peak_hours) if len(peak_hours) > 0 else 'None detected'}")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Binary operation pattern: standby (~{min_power:.3f} kW) vs active (~{max_power:.3f} kW)")
        print(f"   • Stable power factor of {avg_power_factor:.2f} indicates good electrical efficiency")
        print(f"   • {len(peak_hours)} hours of peak operation detected")
        
        if len(peak_hours) > 0:
            peak_energy = self.data[high_power_mask]['power_total'].sum()
            peak_percentage = (peak_energy / total_energy) * 100
            print(f"   • Peak hours consume {peak_percentage:.1f}% of total energy")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   • Consider optimizing peak hour operation for energy savings")
        print(f"   • Monitor power factor consistency for electrical efficiency")
        print(f"   • Implement smart scheduling based on observed usage patterns")
        
    def export_summary_report(self, filename=None):
        """Export analysis summary to text file"""
        if filename is None:
            filename = os.path.join(self.output_dir, f'hvac_analysis_report_{self.style}.txt')
            
        with open(filename, 'w') as f:
            f.write("HVAC POWER USAGE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Style: {self.style}\n")
            f.write(f"Equipment: {self.data['equipment_id'].iloc[0]}\n")
            f.write(f"Data Points: {len(self.data)}\n\n")
            
            # Basic statistics
            stats = self.data['power_total'].describe()
            f.write("POWER CONSUMPTION STATISTICS:\n")
            f.write(f"Total Energy: {self.data['power_total'].sum():.3f} kWh\n")
            f.write(f"Average Power: {stats['mean']:.3f} kW\n")
            f.write(f"Peak Power: {stats['max']:.3f} kW\n")
            f.write(f"Minimum Power: {stats['min']:.3f} kW\n")
            f.write(f"Standard Deviation: {stats['std']:.3f} kW\n\n")
            
            # Hourly summary
            hourly_stats = self.data.groupby('hour')['power_total'].mean()
            f.write("HOURLY AVERAGES:\n")
            for hour, avg_power in hourly_stats.items():
                f.write(f"Hour {hour:2d}: {avg_power:.3f} kW\n")
                
        logging.info(f"Summary report exported to: {filename}")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        logging.info("Starting HVAC Power Usage Analysis")
        print("🚀 Starting HVAC Power Usage Analysis")
        print("=" * 60)
        print(f"📋 Style: {self.style} ({self.get_available_styles()[self.style]})")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"🖼️ Format: {self.output_format} (DPI: {self.dpi})")
        
        # Basic statistics
        self.basic_statistics()
        
        # Hourly analysis
        self.hourly_analysis()
        
        # Pattern detection
        self.detect_patterns()
        
        # Generate insights
        self.generate_insights()
        
        # Create visualizations
        self.create_visualizations()
        self.create_detailed_analysis_plots()
        
        # Export report
        self.export_summary_report()
        
        print(f"\n✅ Analysis complete! Check the '{self.output_dir}' directory for generated files.")
        logging.info("Analysis pipeline completed successfully")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HVAC Power Usage Pattern Analysis Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hvac_analysis.py test.csv
  python hvac_analysis.py test.csv --style talk --format pdf
  python hvac_analysis.py test.csv --output-dir results --dpi 600 --verbose
        """
    )
    
    parser.add_argument(
        'data_file',
        type=str,
        help='Path to HVAC power consumption CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Directory to save plots and reports (default: plots)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for figures (default: png)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures (default: 300)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--style',
        type=str,
        choices=['paper', 'notebook', 'talk', 'poster'],
        default='notebook',
        help='Plot style (default: notebook)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the analysis"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate input file
    if not os.path.exists(args.data_file):
        logging.error(f"Input file not found: {args.data_file}")
        return
    
    try:
        # Initialize analyzer with command line arguments
        analyzer = HVACPowerAnalyzer(
            csv_file=args.data_file,
            style=args.style,
            output_dir=args.output_dir,
            output_format=args.format,
            dpi=args.dpi
        )
        
        # Run complete analysis
        analyzer.run_complete_analysis()
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

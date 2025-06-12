#!/usr/bin/env python3
"""
AC Power Visualization - Configuration and Batch Processing Examples

This file shows how to create configuration presets and run batch analyses
with different styling options for various output formats.
"""

import os
from ac_data_visualization import complete_ac_analysis, visualize_existing_data

# Configuration presets for different use cases
PRESETS = {
    'publication': {
        'style': 'white',
        'context': 'paper', 
        'palette': 'colorblind',
        'fig_format': 'pdf',
        'dpi': 300,
        'font_scale': 0.9,
        'show_plots': False
    },
    
    'presentation': {
        'style': 'darkgrid',
        'context': 'talk',
        'palette': 'bright', 
        'fig_format': 'png',
        'dpi': 150,
        'font_scale': 1.3,
        'show_plots': True
    },
    
    'poster': {
        'style': 'whitegrid',
        'context': 'poster',
        'palette': 'deep',
        'fig_format': 'svg', 
        'dpi': 600,
        'font_scale': 1.5,
        'show_plots': False
    },
    
    'report': {
        'style': 'ticks',
        'context': 'paper',
        'palette': 'muted',
        'fig_format': 'pdf',
        'dpi': 300, 
        'font_scale': 1.1,
        'show_plots': False
    },
    
    'notebook': {
        'style': 'darkgrid',
        'context': 'notebook', 
        'palette': 'husl',
        'fig_format': 'png',
        'dpi': 200,
        'font_scale': 1.0,
        'show_plots': True
    },
    
    'web': {
        'style': 'whitegrid',
        'context': 'notebook',
        'palette': 'Set2',
        'fig_format': 'png',
        'dpi': 150,
        'font_scale': 1.0,
        'show_plots': False
    }
}

def run_analysis_with_preset(preset_name, data_folder="/data/", 
                           output_folder=None, **overrides):
    """
    Run analysis with a predefined preset configuration
    
    Parameters:
    -----------
    preset_name : str
        Name of preset: 'publication', 'presentation', 'poster', 'report', 'notebook', 'web'
    data_folder : str
        Path to data folder
    output_folder : str, optional
        Custom output folder (default: ./output_{preset_name}/)
    **overrides : dict
        Override any preset parameters
    """
    
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    # Get preset configuration
    config = PRESETS[preset_name].copy()
    
    # Apply overrides
    config.update(overrides)
    
    # Set default output folder
    if output_folder is None:
        output_folder = f"./output_{preset_name}/"
    
    print(f"Running AC analysis with '{preset_name}' preset...")
    print(f"Configuration: {config}")
    
    # Run analysis
    unified_df, summary_df = complete_ac_analysis(
        data_folder=data_folder,
        output_folder=output_folder,
        **config
    )
    
    return unified_df, summary_df

def batch_analysis(data_folder="/data/", base_output="./batch_output/"):
    """
    Run analysis with multiple presets for different output formats
    """
    
    print("Running batch analysis with multiple presets...")
    print("=" * 60)
    
    results = {}
    
    for preset_name in PRESETS.keys():
        try:
            output_folder = os.path.join(base_output, preset_name)
            print(f"\nProcessing with {preset_name} preset...")
            
            unified_df, summary_df = run_analysis_with_preset(
                preset_name=preset_name,
                data_folder=data_folder,
                output_folder=output_folder
            )
            
            results[preset_name] = {
                'unified_df': unified_df,
                'summary_df': summary_df,
                'output_folder': output_folder
            }
            
            print(f"✓ {preset_name} preset completed successfully")
            
        except Exception as e:
            print(f"✗ Error with {preset_name} preset: {e}")
            results[preset_name] = None
    
    print(f"\nBatch analysis complete! Results saved in: {base_output}")
    return results

def create_custom_preset(name, **config):
    """
    Create a custom preset configuration
    
    Example:
    create_custom_preset('my_style', 
                        style='dark', 
                        context='talk', 
                        palette='viridis',
                        fig_format='png',
                        dpi=200)
    """
    PRESETS[name] = config
    print(f"Created custom preset '{name}': {config}")

# Example usage functions
def publication_analysis(data_folder="/data/"):
    """Quick function for publication-ready figures"""
    return run_analysis_with_preset('publication', data_folder)

def presentation_analysis(data_folder="/data/"):
    """Quick function for presentation figures"""  
    return run_analysis_with_preset('presentation', data_folder)

def poster_analysis(data_folder="/data/"):
    """Quick function for poster figures"""
    return run_analysis_with_preset('poster', data_folder)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AC Analysis with Configuration Presets')
    parser.add_argument('--preset', choices=list(PRESETS.keys()), 
                       default='notebook', help='Configuration preset to use')
    parser.add_argument('--data-folder', default='/data/', 
                       help='Path to data folder')
    parser.add_argument('--output-folder', 
                       help='Output folder (default: ./output_{preset}/)')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch analysis with all presets')
    parser.add_argument('--list-presets', action='store_true',
                       help='List available presets')
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available presets:")
        print("=" * 30)
        for name, config in PRESETS.items():
            print(f"{name}:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            print()
    
    elif args.batch:
        batch_analysis(args.data_folder)
        
    else:
        run_analysis_with_preset(
            preset_name=args.preset,
            data_folder=args.data_folder, 
            output_folder=args.output_folder
        )

# Example: Create custom configurations
if __name__ == "__main__":
    # Example custom presets
    create_custom_preset('dark_theme',
                        style='dark',
                        context='notebook', 
                        palette='bright',
                        fig_format='png',
                        dpi=200,
                        font_scale=1.1,
                        show_plots=True)
    
    create_custom_preset('scientific',
                        style='white',
                        context='paper',
                        palette='colorblind', 
                        fig_format='eps',
                        dpi=600,
                        font_scale=0.8,
                        show_plots=False)

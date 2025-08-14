import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

# Read the CSV file
df = pd.read_csv('./output/cleaned_Freezer_Main_Service_chart.csv')

# Convert DateTime column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Set DateTime as index for better plotting
df.set_index('DateTime', inplace=True)

# Set up seaborn style for better aesthetics
sns.set_theme(style='whitegrid', palette='muted')
sns.set_context("paper", font_scale=1.2)
# plt.style.use('seaborn-darkgrid')

# Create the line plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Main Service | CT-4 Freezer'], linewidth=1.5, color='blue')

# Customize the plot
plt.title('Freezer Main Service Power Consumption Over A Week', fontsize=14, fontweight='bold')
plt.xlabel('Date and Time', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.grid(True, alpha=0.3)

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Show every 6 hours
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display basic statistics
print("Power Consumption Statistics:")
print(f"Average Power: {df['Main Service | CT-4 Freezer'].mean():.2f} kW")
print(f"Maximum Power: {df['Main Service | CT-4 Freezer'].max():.2f} kW")
print(f"Minimum Power: {df['Main Service | CT-4 Freezer'].min():.2f} kW")
print(f"Standard Deviation: {df['Main Service | CT-4 Freezer'].std():.2f} kW")

# Show the plot
# plt.show()

# Optional: Save the plot as an image
plt.savefig('plots/Freezer/freezer_power_consumption.png', dpi=600, bbox_inches='tight')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import calendar

def analyze_wildfires():
    print("Loading data...")
    fire_df = pd.read_csv('data/wildfire_sizes_before_2010.csv')
    states_df = pd.read_csv('data/merged_state_data.csv')
    weather_df = pd.read_csv('data/weather_monthly_state_aggregates.csv')

    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)

    # Process data
    fire_df['month_num'] = pd.to_datetime(fire_df['month']).dt.month
    fire_df['year'] = pd.to_datetime(fire_df['month']).dt.year
    
    # Merge with weather data
    merged_df = fire_df.merge(weather_df, 
                             left_on=['STATE', 'month'], 
                             right_on=['State', 'year_month'])

    # 1. Fire Size Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=fire_df, x='total_fire_size', bins=50)
    plt.title('Distribution of Fire Sizes')
    plt.xlabel('Fire Size')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('plots/fire_size_distribution.png')
    plt.close()

    # 2. Monthly Patterns
    monthly_avg = fire_df.groupby('month_num')['total_fire_size'].mean()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=monthly_avg.index, y=monthly_avg.values)
    plt.title('Average Fire Size by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Fire Size')
    plt.savefig('plots/monthly_pattern.png')
    plt.close()

    # 3. State Analysis
    state_avg = fire_df.groupby('STATE')['total_fire_size'].agg(['mean', 'count']).round(2)
    state_avg = state_avg.sort_values('mean', ascending=False)

    plt.figure(figsize=(15, 6))
    sns.barplot(x=state_avg.index[:10], y=state_avg['mean'][:10])
    plt.title('Top 10 States by Average Fire Size')
    plt.xticks(rotation=45)
    plt.xlabel('State')
    plt.ylabel('Average Fire Size')
    plt.tight_layout()
    plt.savefig('plots/state_pattern.png')
    plt.close()

    # 4. Time Series Analysis
    yearly_trend = fire_df.groupby('year')['total_fire_size'].agg(['mean', 'sum'])
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(yearly_trend.index, yearly_trend['mean'], marker='o')
    plt.title('Average Fire Size by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Fire Size')
    
    plt.subplot(1, 2, 2)
    plt.plot(yearly_trend.index, yearly_trend['sum'], marker='o', color='red')
    plt.title('Total Fire Size by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Fire Size')
    plt.tight_layout()
    plt.savefig('plots/yearly_trends.png')
    plt.close()

    # 5. Weather Correlation Analysis
    plt.figure(figsize=(12, 8))
    weather_corr = merged_df[['total_fire_size', 'PRCP', 'EVAP', 'TMIN', 'TMAX']].corr()
    sns.heatmap(weather_corr, annot=True, cmap='RdBu_r', center=0)
    plt.title('Correlation between Fire Size and Weather Conditions')
    plt.tight_layout()
    plt.savefig('plots/weather_correlation.png')
    plt.close()

    # 6. Geographic Analysis
    state_total = fire_df.groupby('STATE')['total_fire_size'].sum().reset_index()
    state_data = states_df.merge(state_total, left_on='State', right_on='STATE')
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(state_data['mean_elevation'], state_data['total_fire_size'])
    plt.xlabel('Mean Elevation')
    plt.ylabel('Total Fire Size')
    plt.title('Fire Size vs Elevation')
    
    plt.subplot(1, 2, 2)
    plt.scatter(state_data['Percentage of Federal Land'].str.rstrip('%').astype(float),
                state_data['total_fire_size'])
    plt.xlabel('Percentage of Federal Land')
    plt.ylabel('Total Fire Size')
    plt.title('Fire Size vs Federal Land')
    plt.tight_layout()
    plt.savefig('plots/geographic_analysis.png')
    plt.close()

    # 7. Seasonal Patterns by Region
    regions = {
        'West': ['CA', 'OR', 'WA', 'ID', 'MT', 'WY', 'NV', 'UT', 'CO', 'AZ', 'NM'],
        'South': ['TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'SC', 'NC', 'TN', 'KY'],
        'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'PA', 'NJ', 'MD', 'DE'],
        'Midwest': ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH']
    }
    
    fire_df['Region'] = fire_df['STATE'].map({state: region 
                                             for region, states in regions.items() 
                                             for state in states})
    
    plt.figure(figsize=(15, 8))
    for region in regions:
        region_data = fire_df[fire_df['Region'] == region]
        monthly_avg = region_data.groupby('month_num')['total_fire_size'].mean()
        plt.plot(monthly_avg.index, monthly_avg.values, label=region, marker='o')
    
    plt.title('Seasonal Fire Patterns by Region')
    plt.xlabel('Month')
    plt.ylabel('Average Fire Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/regional_patterns.png')
    plt.close()

    # 8. Correlation Matrix for All Features
    plt.figure(figsize=(15, 12))
    all_features = pd.concat([
        merged_df[['total_fire_size', 'PRCP', 'EVAP', 'TMIN', 'TMAX']],
        pd.get_dummies(merged_df['Region'], prefix='region'),
        states_df.set_index('State').loc[merged_df['STATE']][['mean_elevation', 'Land Area (sq mi)', 
                                                             'Percentage of Federal Land', 'Urbanization Rate (%)']]
    ], axis=1)
    
    corr_matrix = all_features.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Matrix of All Features')
    plt.tight_layout()
    plt.savefig('plots/full_correlation_matrix.png')
    plt.close()

    # 9. Fire Size Distribution by Region (Violin Plot)
    plt.figure(figsize=(15, 8))
    sns.violinplot(data=fire_df, x='Region', y='total_fire_size')
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title('Fire Size Distribution by Region')
    plt.tight_layout()
    plt.savefig('plots/regional_distribution.png')
    plt.close()

    # 10. Year-over-Year Monthly Trends
    pivot_data = fire_df.pivot_table(
        values='total_fire_size',
        index='month_num',
        columns='year',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Average Fire Size by Month and Year')
    plt.xlabel('Year')
    plt.ylabel('Month')
    plt.yticks(range(12), calendar.month_abbr[1:13])
    plt.tight_layout()
    plt.savefig('plots/monthly_yearly_heatmap.png')
    plt.close()

    # 11. Fire Frequency vs Size
    plt.figure(figsize=(12, 6))
    fire_counts = fire_df.groupby('STATE').size()
    fire_sizes = fire_df.groupby('STATE')['total_fire_size'].mean()
    plt.scatter(fire_counts, fire_sizes)
    for state in fire_df['STATE'].unique():
        plt.annotate(state, (fire_counts[state], fire_sizes[state]))
    plt.xlabel('Number of Fire Incidents')
    plt.ylabel('Average Fire Size')
    plt.title('Fire Frequency vs Average Size by State')
    plt.tight_layout()
    plt.savefig('plots/frequency_vs_size.png')
    plt.close()

    # 12. Environmental Factors Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Environmental Factors Impact on Fire Size')
    
    sns.scatterplot(data=merged_df, x='PRCP', y='total_fire_size', ax=axes[0,0])
    axes[0,0].set_title('Precipitation vs Fire Size')
    axes[0,0].set_yscale('log')
    
    sns.scatterplot(data=merged_df, x='TMAX', y='total_fire_size', ax=axes[0,1])
    axes[0,1].set_title('Maximum Temperature vs Fire Size')
    axes[0,1].set_yscale('log')
    
    sns.scatterplot(data=merged_df, x='EVAP', y='total_fire_size', ax=axes[1,0])
    axes[1,0].set_title('Evaporation vs Fire Size')
    axes[1,0].set_yscale('log')
    
    sns.scatterplot(data=merged_df, x='TMIN', y='total_fire_size', ax=axes[1,1])
    axes[1,1].set_title('Minimum Temperature vs Fire Size')
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/environmental_factors.png')
    plt.close()

    # Print statistics
    print("\nOverall Fire Statistics:")
    print(fire_df['total_fire_size'].describe().round(2))
    
    print("\nTop 5 States by Average Fire Size:")
    print(state_avg[['mean', 'count']].head())
    
    print("\nWorst Months for Fires:")
    print(monthly_avg.sort_values(ascending=False).head())

    # Print additional statistics
    print("\nRegional Analysis:")
    regional_stats = fire_df.groupby('Region')['total_fire_size'].agg(['mean', 'sum', 'count'])
    print(regional_stats.round(2))
    
    print("\nWeather Correlation Summary:")
    for column in ['PRCP', 'EVAP', 'TMIN', 'TMAX']:
        correlation = stats.pearsonr(merged_df['total_fire_size'], merged_df[column])
        print(f"{column}: correlation={correlation[0]:.3f}, p-value={correlation[1]:.3f}")

    # Additional Statistics
    print("\nTop 5 Most Severe Fire Months (Year-Month):")
    monthly_totals = fire_df.groupby(['year', 'month_num'])['total_fire_size'].sum().sort_values(ascending=False)
    print(monthly_totals.head())
    
    print("\nYear-over-Year Growth Rate:")
    yearly_totals = fire_df.groupby('year')['total_fire_size'].sum()
    growth_rates = (yearly_totals - yearly_totals.shift(1)) / yearly_totals.shift(1) * 100
    print(growth_rates)
    
    print("\nSeasonal Severity (Average fire size by season):")
    fire_df['season'] = pd.cut(fire_df['month_num'], 
                              bins=[0, 3, 6, 9, 12], 
                              labels=['Winter', 'Spring', 'Summer', 'Fall'])
    seasonal_avg = fire_df.groupby('season')['total_fire_size'].mean().sort_values(ascending=False)
    print(seasonal_avg)

if __name__ == "__main__":
    analyze_wildfires()

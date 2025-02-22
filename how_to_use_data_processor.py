from src.data_processor import WildfireData

data = WildfireData(
    state_data_path="data/merged_state_data.csv",
    weather_data_path="data/weather_monthly_state_aggregates.csv",
    fire_data_path="data/wildfire_sizes_before_2010.csv"
)
data.prepare_data()

print(data.X_train.columns)
print(data.X_val.head())
print(data.X_test.head())
print(data.y_train.columns)
print(data.y_val.columns)

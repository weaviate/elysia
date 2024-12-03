import random
import pandas as pd
import json
from tqdm.auto import tqdm
from datetime import datetime, timedelta

class WeatherDataGenerator:
    def __init__(self, num_days=50):
        self.num_days = num_days

    def generate_random_date(self, start_date):
        return start_date + timedelta(days=random.randint(0, 365))

    def generate_weather_data(self):
        start_date = datetime(2023, 1, 1)
        weather_data = []

        for _ in tqdm(range(self.num_days), desc="Generating weather data"):
            date = self.generate_random_date(start_date)
            temperature = round(random.uniform(-10, 35), 1)  # Temperature in °C
            humidity = random.randint(20, 100)  # Humidity in %
            wind_speed = round(random.uniform(0, 100), 1)  # Wind speed in km/h
            precipitation = round(random.uniform(0, 50), 1)  # Precipitation in mm
            pressure = round(random.uniform(950, 1050), 1)  # Atmospheric pressure in hPa
            visibility = round(random.uniform(1, 10), 1)  # Visibility in km

            weather_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "precipitation": precipitation,
                "pressure": pressure,
                "visibility": visibility
            })

        return weather_data

if __name__ == "__main__":
    generator = WeatherDataGenerator(num_days=50)
    weather_data = generator.generate_weather_data()

    with open("weather_data.jsonl", "w") as f:
        for data in weather_data:
            json.dump(data, f)
            f.write("\n")

    weather_df = pd.DataFrame(weather_data)
    weather_df.to_csv("weather_data.csv", index=False)

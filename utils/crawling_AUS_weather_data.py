import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Initialize Selenium WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Set date range for data collection
location_name = "Yulara"

if location_name == "Alice_Springs":
    location_code = 15590
    start_date = datetime(2008, 1, 1)
elif location_name == "Yulara":
    location_code = 15635
    # start_date = datetime(2016, 1, 1)
    start_date = datetime(2024, 1, 1)

end_date = datetime(2024, 11, 11)
current_date = start_date
total_days = (end_date - start_date).days + 1

# List to store failed dates
failed_dates = []

# Dictionary to store yearly data
yearly_data = {}

# Loop through each date to collect data
for _ in tqdm(range(total_days), desc="Progress", unit="day"):
    date_str = current_date.strftime('%Y-%m-%d')
    url = f'https://www.weatherzone.com.au/station/SITE/{location_code}/observations/{date_str}'
    driver.get(url)

    try:
        # Wait until the table is loaded
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'tbody'))
        )

        # Get page source once the JavaScript content is loaded
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Look for the table without relying on dynamically changing classes
        table_body = soup.find('tbody')
        if table_body:
            rows = table_body.find_all('tr')[1:]  # Skip the first row with the next day's data
            daily_data = []

            for row in rows:
                time_tag = row.find('td', class_='hourly-obs-date')
                humidity_tag = row.find('td', class_='hourly-obs-humidityt')
                wind_speed_tag = row.find('td', class_='hourly-obs-windSpeed')

                if time_tag and humidity_tag and wind_speed_tag:
                    time_str = time_tag.get_text(strip=True)
                    time_str = " ".join(time_str.split()[1:])  # Get time only, skipping weekday
                    time_str = time_str.replace(" ACST", "").replace(" ACDT", "")  # Remove timezone
                    try:
                        timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                        humidity_str = humidity_tag.get_text(strip=True)
                        wind_speed_str = wind_speed_tag.get_text(strip=True)
                        if timestamp.date() == current_date.date():  # Exclude next day's data
                            daily_data.append([timestamp, humidity_str, wind_speed_str])
                    except ValueError as ve:
                        print(f"Failed to parse time for {date_str}: {ve}")
                        failed_dates.append(f"{date_str} {time_str}")

            # Sort daily data in ascending order by timestamp
            daily_data.sort(key=lambda x: x[0])

            # Add daily data to yearly data dictionary
            year = current_date.year
            if year not in yearly_data:
                yearly_data[year] = []
            yearly_data[year].extend(daily_data)
        else:
            print(f"No data table found for {date_str}")
            failed_dates.append(date_str)

    except Exception as e:
        print(f"Failed to retrieve data for {date_str}: {e}")
        failed_dates.append(date_str)

    # Move to the next day
    current_date += timedelta(days=1)

    # Save accumulated data at the end of each year
    if current_date.year != year:
        # with open(f'{location_name}_weather_data_{year}.csv', mode='w', newline='') as file:
        with open(f'{location_name}_weather_data_{year}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'Weather_Relative_Humidity', 'Wind_Speed'])
            writer.writerows(yearly_data[year])
        del yearly_data[year]
    # Save data for the last year if not already saved
    elif current_date.year in yearly_data:
        with open(f'{location_name}_weather_data_{current_date.year}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'Weather_Relative_Humidity', 'Wind_Speed'])
            writer.writerows(yearly_data[current_date.year])

# Close the driver
driver.quit()

# Print failed dates
if failed_dates:
    print("\nFailed to retrieve data for the following timestamps:")
    for date in failed_dates:
        print(date)

print("Data collection complete.")
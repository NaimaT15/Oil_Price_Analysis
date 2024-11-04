import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# Function to scrape OPEC press releases for a specific year
def scrape_opec_year(year):
    url = f'https://www.investopedia.com/ask/answers/012715/what-causes-oil-prices-fluctuate.asp'  # URL format based on year
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve data for {year} (Status Code: {response.status_code})")
        return pd.DataFrame()  # Return an empty DataFrame if the page cannot be accessed

    # Print response content for inspection
    print(f"Inspecting content for {year}...")
    print(response.text[:500])  # Print the first 500 characters of the response for a quick check
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Initialize a list to hold OPEC event data
    opec_events = []

    # Find all relevant press release items
    for release in soup.find_all('div', class_='press_item'):
        date_str = release.find('div', class_='date').get_text(strip=True)
        title = release.find('h3').get_text(strip=True)
        description = release.find('p').get_text(strip=True)

        # Parse the date
        try:
            date = datetime.strptime(date_str, '%d %B %Y')  # Adjust the format based on actual date string
        except ValueError:
            print(f"Date parsing failed for year {year}: {date_str}")
            continue

        # Append only if date, title, and description are found
        if date and title and description:
            opec_events.append({
                'Date': date,
                'Event Type': 'OPEC Announcement',
                'Description': title + ": " + description
            })

    # Create a DataFrame from the list of events
    return pd.DataFrame(opec_events)

import pandas as pd
import numpy as np

# Load the improved CSV files
improved_health_data = pd.read_csv('improved_health_data.csv')
improved_sentiment_data = pd.read_csv('improved_sentiment_data.csv')
improved_recommendation_data = pd.read_csv('improved_recommendation_data.csv')
improved_traffic_data = pd.read_csv('improved_traffic_data.csv')

# Function to add new columns for more variety
def add_columns_health_data(df):
    df['hospital_access'] = np.random.randint(0, 2, df.shape[0])
    df['vaccination_rate'] = np.random.uniform(0, 1, df.shape[0])
    return df

def add_columns_sentiment_data(df):
    df['sentiment'] = df['sentiment'].str.lower()
    return df

def add_columns_recommendation_data(df):
    df['rating'] = np.random.randint(1, 6, df.shape[0])
    df['frequency'] = np.random.randint(1, 10, df.shape[0])
    return df

def add_columns_traffic_data(df):
    df['road_condition'] = np.random.choice(['Good', 'Moderate', 'Bad'], df.shape[0])
    df['accidents'] = np.random.randint(0, 5, df.shape[0])
    return df

# Function to expand recommendation data with new services, interests, and synthetic users
def expand_services_and_interests(df, num_users=15):
    new_services = ['Eco Fair', 'Digital Expo', 'AI Bootcamp', 'Green Market']
    new_interests = ['Sustainability', 'Innovation', 'AI', 'Environment']
    synthetic_rows = []
    for i in range(num_users):
        user = f"SyntheticUser{i+1}"
        sampled_services = np.random.choice(new_services, size=np.random.randint(1, 4), replace=False)
        for service in sampled_services:
            interest = np.random.choice(new_interests)
            rating = np.random.randint(1, 6)
            frequency = np.random.randint(1, 10)
            synthetic_rows.append({
                'user': user,
                'interest': interest,
                'service': service,
                'rating': rating,
                'frequency': frequency
            })
    return pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)

# Apply the functions to add columns
improved_health_data = add_columns_health_data(improved_health_data)
improved_sentiment_data = add_columns_sentiment_data(improved_sentiment_data)
improved_recommendation_data = add_columns_recommendation_data(improved_recommendation_data)
improved_traffic_data = add_columns_traffic_data(improved_traffic_data)

# Expand recommendation data
improved_recommendation_data = expand_services_and_interests(improved_recommendation_data, num_users=15)

# Save the improved CSV files
improved_health_data.to_csv('improved_health_data.csv', index=False)
improved_sentiment_data.to_csv('improved_sentiment_data.csv', index=False)
improved_recommendation_data.to_csv('improved_recommendation_data.csv', index=False)
improved_traffic_data.to_csv('improved_traffic_data.csv', index=False)

print("✅ Improved CSV files have been generated and saved successfully.")
import numpy as np
import pandas as pd

# Generate random data
n_samples = 2000
homes = np.arange(n_samples) + 1
prices = np.random.randint(50000, 500000, size=n_samples)
sqfts = np.random.randint(500, 5000, size=n_samples)
bedrooms = np.random.randint(1, 6, size=n_samples)
bathrooms = np.random.randint(1, 5, size=n_samples)
offers = np.random.randint(0, 6, size=n_samples)
bricks = np.random.choice(['Yes', 'No'], size=n_samples)
neighborhoods = np.random.choice(['East', 'West', 'North', 'South'], size=n_samples)

# Create dataframe
df = pd.DataFrame({
    'Home': homes,
    'Price': prices,
    'SqFt': sqfts,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Offers': offers,
    'Brick': bricks,
    'Neighborhood': neighborhoods
})

# Preview the data
print(df.head())

# Save the data to a CSV file
with open('house-price-random.csv', 'w') as f:
    df.to_csv(f, index=False)


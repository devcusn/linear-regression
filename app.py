import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv('./data/average_hourly_earnings_of_female_and_male_employees.csv')

df.sort_values(by='year', inplace=True)

turkiye_data = df[df['country'] == 'Türkiye']

female_data = turkiye_data[turkiye_data['gender'] == 'Female']
male_data = turkiye_data[turkiye_data['gender'] == 'Male']

X_female = female_data['year'].values.reshape(-1, 1)
y_female = female_data['amount_local_currency'].values
model_female = LinearRegression()
model_female.fit(X_female, y_female)


X_male = male_data['year'].values.reshape(-1, 1)
y_male = male_data['amount_local_currency'].values
model_male = LinearRegression()
model_male.fit(X_male, y_male)

years_to_predict = np.arange(2010, 2030).reshape(-1, 1)

predicted_values_female = model_female.predict(years_to_predict)


predicted_values_male = model_male.predict(years_to_predict)


predictions_df = pd.DataFrame({
    'Year': years_to_predict.flatten(),
    'Female_Predicted': predicted_values_female,
    'Male_Predicted': predicted_values_male
})

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(female_data['year'], female_data['amount_local_currency'], label='Female Predictions', color='yellow')
plt.plot(predictions_df['Year'], predictions_df['Female_Predicted'], label='Female Predictions', color='blue')
plt.xlabel('Year')
plt.ylabel('Amount in Local Currency')
plt.title('Turkey Female Predictions (2013-2100)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(male_data['year'], male_data['amount_local_currency'], label='Male Predictions', color='orange')
plt.plot(predictions_df['Year'], predictions_df['Male_Predicted'], label='Male Predictions', color='green')
plt.xlabel('Year')
plt.ylabel('Amount in Local Currency')
plt.title('Turkey Male Predictions (2013-2100)')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()

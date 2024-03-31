import numpy as np

# Example data simulating ransomware incidents per state
data = {
    "State": ["California", "Texas", "Florida", "New York", "Illinois", "Pennsylvania", "Ohio", "Michigan", 
              "Georgia", "North Carolina", "Virginia", "New Jersey", "Washington", "Arizona", "Massachusetts", 
              "Tennessee", "Indiana"],
    "Incidents": [120, 95, 80, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
}

# Convert to DataFrame
df_example = pd.DataFrame(data)

# For a simple linear regression analysis, we will encode the states as numeric values (although in this context, it's a bit artificial)
df_example['StateID'] = range(len(df_example['State']))

# Perform a simple linear regression
from sklearn.linear_model import LinearRegression

X = df_example['StateID'].values.reshape(-1, 1)
y = df_example['Incidents'].values

model = LinearRegression()
model.fit(X, y)

# Predict values for the regression line
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(df_example['StateID'], df_example['Incidents'], color='blue', label='Actual Incidents')
plt.plot(df_example['StateID'], y_pred, color='red', label='Regression Line')
plt.title('Ransomware Incidents by State in 2021')
plt.xlabel('State (Encoded as ID)')
plt.ylabel('Number of Ransomware Incidents')
plt.xticks(ticks=df_example['StateID'], labels=df_example['State'], rotation='vertical')
plt.legend()
plt.tight_layout()
plt.show()

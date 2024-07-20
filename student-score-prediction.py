import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)

# Visualize the data
plt.scatter(data['Hours'], data['Scores'])
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

# Prepare the data
X = data[['Hours']].values
y = data['Scores'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Predict the score for 9.25 hours/day
hours = np.array([[9.25]])
predicted_score = regressor.predict(hours)
print(f"Predicted score for a student studying 9.25 hours/day: {predicted_score[0]}")

# Plot the regression line
plt.scatter(X, y)
plt.plot(X, regressor.coef_ * X + regressor.intercept_, color='red')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

import pandas as pd
from flask import Flask, request, render_template
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
# Load the input data
data = pd.read_csv('D:\Study\FPTU\Season4\ADY201m\ADY201m_SP23_AI17A\Worklab\Lab4\Polynomial_linear_regression\house-prices.csv')

# Convert the integer feature names to strings
data.columns = data.columns.astype(str)

# Perform preprocessing
data['Neighborhood'] = data['Neighborhood'].astype('category')
data = pd.get_dummies(data, columns=['Neighborhood'], prefix='', prefix_sep='')

data['Brick'] = data['Brick'].str.replace('Yes', '1').str.replace('No', '0').astype(int)

# Split the data into training and testing sets

X = data[['SqFt', 'Bedrooms', 'Bathrooms', 'Offers', 'Brick', 'West', 'East', 'North', 'South']]
y = data['Price']

model = LinearRegression()
model.fit(X, y)

features = [2000, 4, 2, 3, 1, 0, 1, 0, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_pred = model.predict(features.transform(X_test))

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)
print('R-squared score:', r2)

# Deploy the model in Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the price of a house based on user input
    """
    # Get the user input
    features = []
    # Convert numeric features to float
    for x in request.form.values():
        if x != 'Yes' and x != 'No' and x != 'West' and x != 'East' and x != 'North' and x != 'South':
            features.append(float(x))
    # Convert the neighborhood feature to one-hot encoding
    neighborhood = request.form.get('Neighborhood', default='')
    neighborhoods = ['West', 'East', 'North', 'South']
    neighborhood_feature = [1 if neighborhood == n else 0 for n in neighborhoods]
    # Convert the brick feature to one-hot encoding
    brick = request.form.get('Brick', default='')
    brick_feature = [1 if brick == 'Yes' else 0]
    # Combine the user input with the neighborhood feature
    features = features + brick_feature + neighborhood_feature
    # Make a prediction
    prediction = model.predict([features])
    # Format the prediction as currency
    prediction = '${:,.2f}'.format(prediction[0])
    # Render the HTML template with the prediction
    return render_template('result.html', prediction_text='Predicted price: {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)


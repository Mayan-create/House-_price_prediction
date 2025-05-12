import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to generate synthetic house data
def generate_house_data(n_samples=100):
    size = np.random.randint(500, 3500, size=n_samples)
    price = size * 100 + np.random.randint(-10000, 10000, size=n_samples)
    return pd.DataFrame({'size': size, 'price': price})

# Function to train the model
def train_model(df):
    X = df[['size']]
    Y = df[['price']]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

# Streamlit app
def main():
    st.title('🏠 House Price Prediction')
    st.write('This app predicts the price of a house based on its size (in square feet).')

    # Generate and show data
    df = generate_house_data(n_samples=100)
    model = train_model(df)

    # User input
    size = st.number_input('Enter the size of the house (sq ft):', min_value=100, max_value=3500, value=1000)

    if st.button("Predict Price"):
        predicted_price = model.predict([[size]])
        st.success(f'The predicted price of the house is **${predicted_price[0][0]:,.2f}**')

        # Plotting
        fig, ax = plt.subplots()
        ax.scatter(df['size'], df['price'], label='Actual Prices', alpha=0.6)
        ax.plot(df['size'], model.predict(df[['size']]), color='red', label='Regression Line')
        ax.set_xlabel('Size (sq ft)')
        ax.set_ylabel('Price (Ruppe)')
        ax.set_title('House Price vs Size')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()

    
 
  
                    
  


     

    
  

   

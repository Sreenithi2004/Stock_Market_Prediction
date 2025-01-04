import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
import streamlit as st
from datetime import datetime

# Path to the JSON file storing user data
USER_DATA_FILE = "user_data.json"

# Load user data from JSON file
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {}

# Save user data to JSON file
def save_user_data(user_data):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(user_data, file, indent=4)

# Initialize user data
USER_DATA = load_user_data()

# Validation functions
def is_valid_username(username):
    if not username:
        return False, "Username is required."
    if username[0].isdigit():
        return False, "Username must not start with a digit."
    if len(username) < 6:
        return False, "Username must be at least 6 characters long."
    return True, ""

def is_valid_password(password):
    if not password:
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character."
    return True, ""

# Authentication functions
def login(username, password):
    if username in USER_DATA and USER_DATA[username] == password:
        return True
    return False

def signup(username, password):
    if username in USER_DATA:
        return False  # Username already exists
    USER_DATA[username] = password
    save_user_data(USER_DATA)  # Save new user data
    return True

# Manage session state for user authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Login/Signup Page
if not st.session_state.logged_in:
    st.title("Welcome to Stock Price Prediction App")

    choice = st.radio("Choose an option", ["Login", "Signup"])

    # Input fields with validation checks
    username = st.text_input("Username")
    is_username_valid, username_message = is_valid_username(username)
    if username and not is_username_valid:
        st.error(username_message)

    password = st.text_input("Password", type="password")
    is_password_valid, password_message = is_valid_password(password)
    if password and not is_password_valid:
        st.error(password_message)

    if choice == "Login":
        if st.button("Login"):
            if is_username_valid and is_password_valid:
                if login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome {username}!")
                    st.rerun()  # Redirect to dashboard
                else:
                    st.error("Invalid username or password.")
            else:
                st.error("Please fix the input errors above.")
    elif choice == "Signup":
        if st.button("Signup"):
            if is_username_valid and is_password_valid:
                if signup(username, password):
                    st.success("Account created successfully! Please log in.")
                else:
                    st.error("Username already exists.")
            else:
                st.error("Please fix the input errors above.")

# Dashboard (Main Application after login)
if st.session_state.logged_in:
    st.title("Stock Price Prediction Dashboard")
    st.write(f"Logged in as: {st.session_state.username}")

    # Sidebar for user inputs
    st.sidebar.header("User Inputs")

    # Predefined list of stock symbols for dropdown
    STOCK_SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "META", "NFLX", "NVDA", "JPM", "V",
        "JNJ", "WMT", "DIS", "BAC", "PG"
    ]

    # Dropdown for stock selection
    stock_symbol = st.sidebar.selectbox("Select Stock Symbol", STOCK_SYMBOLS)

    # Input for start date
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))

    # Predict button under inputs
    

    if st.sidebar.button("Predict"):
        # Function to get stock data for the next 7 days from the given start date
        def get_stock_data(symbol, start_date):
            end_date = start_date + pd.Timedelta(days=7)
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            return stock_data

        # Function to prepare data for Multilinear Regression
        def prepare_data(stock_data):
            stock_data['Date'] = stock_data.index
            stock_data['Date'] = pd.to_numeric(stock_data['Date'])
            X = stock_data[['Date', 'Open']]
            y = stock_data['Close']
            return X, y

        # Function to train Multilinear Regression model
        def train_model(X, y):
            if len(X) < 2:  # Check if we have enough data to split into train and test sets
                return None, None, None, None, None
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            return model, mse, X_test, y_test, y_pred

        # Load and display stock data
        stock_data = get_stock_data(stock_symbol, start_date)
        if not stock_data.empty:
            st.write(f"Showing stock data for {stock_symbol} from {start_date} for 7 days")
            st.dataframe(stock_data.tail())

            # Prepare the data for the model
            X, y = prepare_data(stock_data)

            # Train the model and display results
            model, mse, X_test, y_test, y_pred = train_model(X, y)

            # Plot the predictions
            st.subheader("Stock Price Prediction (7 Days)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(stock_data.index, stock_data['Close'], color='blue', label='Actual Prices')
            ax.plot(stock_data.index, model.predict(X), color='red', label='Predicted Prices')
            ax.set_title(f"{stock_symbol} Stock Price Prediction (7 Days)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")
            ax.legend()
            st.pyplot(fig)

            # Display Mean Squared Error
            st.write(f"Mean Squared Error of the model: {mse:.2f}")
        else:
            st.error("Failed to retrieve stock data. Please check the stock symbol or start date.")

    # Logout option
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()  # Redirect back to login

#hey

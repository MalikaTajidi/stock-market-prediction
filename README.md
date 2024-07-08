
# Apple Stock Price Analysis and Prediction

This project analyzes and predicts Apple stock prices using historical data. The analysis includes data visualization and linear regression for predicting stock prices.

## Project Structure

- `AAPL.csv`: CSV file containing historical stock price data for Apple.
- `analysis.py`: Python script for data analysis and visualization.
- `README.md`: Project documentation.

## Getting Started

### Prerequisites

Ensure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- plotly
- scikit-learn

You can install these using pip:

```bash
pip install pandas numpy matplotlib plotly scikit-learn
```

### Running the Analysis

1. **Load the Data**: Read the CSV file containing Apple stock prices.
    ```python
    import pandas as pd
    apple = pd.read_csv('C:\\Users\\HP\\Desktop\\AAPL.csv')
    ```

2. **Data Preprocessing**: Convert the 'Date' column to datetime format and explore the dataset.
    ```python
    apple['Date'] = pd.to_datetime(apple['Date'])
    print(f'dataframe contains stock prices between {apple.Date.min()} {apple.Date.max()}')
    print(f'total days {(apple.Date.max() - apple.Date.min()).days}')
    apple.describe()
    ```

3. **Data Visualization**: Create box plots for stock prices and line plots for the closing prices.
    ```python
    import matplotlib.pyplot as plt
    import plotly.graph_objs as go
    from plotly.offline import plot, iplot

    apple[['Open','High','Low','Close','Adj Close']].plot(kind='box')

    layout = go.Layout(
        title='Stock Prices of Apple',
        xaxis=dict(
            title='Date',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color="#1f77b4" 
            )
        ),
        yaxis=dict(
            title='Price',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color="#1f77b4"  
            )
        )
    )
    apple_data = [{'x': apple['Date'], 'y': apple['Close']}]
    plot = go.Figure(data=apple_data, layout=layout)
    iplot(plot)
    ```

4. **Train-Test Split**: Split the data into training and testing sets.
    ```python
    from sklearn.model_selection import train_test_split
    X = np.array(apple.index).reshape(-1, 1)
    Y = apple['Close']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
    ```

5. **Standardization**: Standardize the features.
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    ```

6. **Linear Regression**: Train a linear regression model and visualize the results.
    ```python
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train, Y_train)

    trace0 = go.Scatter(
      x=X_train.T[0],
      y=Y_train,
      mode='markers',
      name='Actual'
    )
    trace1 = go.Scatter(
      x=X_train.T[0],
      y=lm.predict(X_train).T,
      mode='lines',
      name='Predicted'
    )
    apple_data = [trace0, trace1]
    layout.xaxis.title.text = 'Day'
    plot2 = go.Figure(data=apple_data, layout=layout)
    iplot(plot2)
    ```

7. **Model Evaluation**: Evaluate the model's performance using R² score and Mean Squared Error (MSE).
    ```python
    from sklearn.metrics import mean_squared_error as mse, r2_score

    scores = f"""
    {'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
    {'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
    {'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
    """
    print(scores)
    ```

## Results

The results section includes plots of the actual vs. predicted stock prices and the evaluation metrics of the linear regression model. 



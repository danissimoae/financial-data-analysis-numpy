import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FinancialDataAnalyzer:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        self.preprocessed_data = self.preprocess_data(self.data)
        self.mean_prices = None
        self.volatilities = None
        self.correlations = None
        self.optimal_weights = None

    def load_data(self, file_path):
        """Load data from csv file"""
        return pd.read_csv(file_path)

    def preprocess_data(self, data):
        """Preprocessing data"""
        if 'Date' not in data.columns:
            raise KeyError("The 'Date' column is missing in the data.")

        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return data

    def calculate_indicators(self):
        """Calculate indicators"""
        prices = self.preprocessed_data.values
        self.mean_prices = np.mean(prices, axis=0)
        self.volatilities = np.std(prices, axis=0)
        self.correlations = np.corrcoef(prices.T)
        return self.mean_prices, self.volatilities, self.correlations

    def optimize_portfolio(self, target_return):
        """Simplest portfolio optimizing model"""
        prices = self.preprocessed_data.values
        cov_matrix = np.cov(prices.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.ones(len(prices.T))
        self.optimal_weights = np.dot(inv_cov_matrix, ones) / np.dot(ones.T, np.dot(inv_cov_matrix, ones))
        return self.optimal_weights

    def plot_data(self):
        """Visualize data and indicators"""
        if self.mean_prices is None or self.volatilities is None or self.correlations is None:
            self.calculate_indicators()

        plt.figure(figsize=(10, 6))
        for column in self.preprocessed_data.columns:
            plt.plot(self.preprocessed_data.index, self.preprocessed_data[column], label=column)
        plt.title('Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.bar(self.preprocessed_data.columns, self.mean_prices)
        plt.title('Mean Stock Prices')
        plt.xlabel('Stocks')
        plt.ylabel('Mean Price')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.bar(self.preprocessed_data.columns, self.volatilities)
        plt.title('Stock Volatilities')
        plt.xlabel('Stocks')
        plt.ylabel('Volatility')
        plt.show()

    def get_recommendations(self):
        """Generate investment recommendations based on indicators"""
        if self.optimal_weights is None:
            print("Optimize portfolio first.")
            return

        recommendations = {}
        for stock, weight in zip(self.preprocessed_data.columns, self.optimal_weights):
            recommendations[stock] = weight

        return recommendations


# Пример использования
analyzer = FinancialDataAnalyzer('stock_prices.csv')
analyzer.calculate_indicators()
analyzer.optimize_portfolio(target_return=0.02)
analyzer.plot_data()
recommendations = analyzer.get_recommendations()
print("Investment Recommendations:", recommendations)

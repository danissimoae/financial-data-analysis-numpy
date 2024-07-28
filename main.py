from analysis_objects import FinancialDataAnalyzer

analyzer = FinancialDataAnalyzer('stock_prices.csv')
analyzer.calculate_indicators()
analyzer.optimize_portfolio(target_return=0.02)
analyzer.plot_data()
recommendations = analyzer.get_recommendations()
print("Инвестиционные рекомендации:", recommendations)
import pandas as pd
import numpy as np

# Создание искусственных данных
dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='B')  # рабочие дни
num_dates = len(dates)

# Генерация случайных данных для цен акций
np.random.seed(0)  # для воспроизводимости
prices = {
    'Date': dates,
    'Company_A': np.random.normal(loc=100, scale=10, size=num_dates).cumsum(),
    'Company_B': np.random.normal(loc=200, scale=20, size=num_dates).cumsum(),
    'Company_C': np.random.normal(loc=300, scale=30, size=num_dates).cumsum(),
    'Company_D': np.random.normal(loc=400, scale=40, size=num_dates).cumsum()
}

# Создание DataFrame
df = pd.DataFrame(prices)
df = df.abs()  # все значения положительные

# Сохранение данных в CSV файл
df.to_csv('stock_prices.csv', index=False)

print(df.head())

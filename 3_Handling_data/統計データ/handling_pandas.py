import pandas as pd

df = pd.read_csv("iris.csv")

y_name = "category"
y = df[y_name].values
x_table = df.drop(y_name, axis = 1)
x_name = x_table.columns
x = x_table.values

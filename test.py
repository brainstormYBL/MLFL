from Models.Models import LinearRegression

model = LinearRegression(10, 20)
a = model.state_dict()
b = a.keys()
bb = 10
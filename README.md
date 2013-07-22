
This is a poisson generalized linear model. It is useful for regression onto count data.

CountRegression.py coutains the PoissonRegression class.

Usage.py shows an example of how to use the code, which is summarized below

model = CountRegression.PoissonRegression(X_train,Y_train)

model.fit()

model.predict(X_test)

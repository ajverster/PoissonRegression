import numpy as np
import matplotlib.pyplot as plt
import random
import CountRegression

#Create poisson data, and noisy predictors
sd_noise = 0.25
poisson_mean = 2.0
sample_size = 100
Y = np.random.poisson(poisson_mean,sample_size)
X1 = 3 * Y + 2 + np.random.normal(0,sd_noise,sample_size)
X2 = 7 * Y + 5 + np.random.normal(0,sd_noise,sample_size)
X = np.matrix(np.vstack((X1,X2)))
X = X.transpose()

#Fit model and test the correlation coefficient
model = CountRegression.PoissonRegression(X,Y)
model.fit()
model.PlotConvergence()
np.corrcoef(np.squeeze(model.predict()),np.squeeze(np.asarray(Y)))[0,1]

plt.scatter(Y,model.predict())
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

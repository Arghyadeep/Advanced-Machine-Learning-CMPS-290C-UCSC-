from sklearn.externals import joblib
import matplotlib.pyplot as plt
Y = joblib.load("Y.pkl")
labels = joblib.load("labels.pkl")
Xs = []
Ys = []
for i in range(len(Y)):
    Xs.append(Y[i][0])
    Ys.append(Y[i][1])

plt.scatter(Xs,Ys,10,labels)
plt.show()

from sklearn.externals import joblib
import matplotlib.pyplot as plt

trustworthiness = joblib.load("trustworthiness.pkl")
continuity = joblib.load("tsne_continuity.pkl")

plt.plot(trustworthiness,'r')
plt.plot(continuity,'b')
plt.legend("trustworthiness",'r')
plt.show()

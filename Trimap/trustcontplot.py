from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

trustworthiness = joblib.load("trimap_trustworthiness.pkl")
continuity = joblib.load("trimap_continuity.pkl")

red_patch = mpatches.Patch(color='red', label='trustworthiness')
blue_patch = mpatches.Patch(color='blue', label='continuity')
plt.legend(handles=[blue_patch,red_patch], loc = 'lower right')
#plt.legend(handles=[red_patch],loc = 'upper right')
plt.legend(loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.plot(trustworthiness,'r')
plt.plot(continuity,'b')
#plt.legend("trustworthiness",'r')
plt.show()

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

list = [100, 200, 300, 400, 500]
x = np.array(list)
y = np.array([0.6190476190476191, 0.8285714285714286, 0.7058823529411765, 0.627906976744186, 0.725])
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')

print type(x), x
print type(y), y

xnew = np.linspace(list[0], list[len(list)-1], num=41, endpoint=True)
print type(xnew), xnew

plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
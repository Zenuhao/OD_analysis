import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Label = np.array([
"Scmr",
"Stdt",
"Syfp",
"Snp",
"Ctdt",
"Cyfp",
"Cnp",
"Rcmr",
"Rtdt",
"Ryfp",
"Rnp"
])

Mean = np.array([
0.000279295,
0.000279886,
0.000378883,
0.000404341,
0.000316444,
0.000351304,
0.000380095,
0.000214262,
0.000246574,
0.000313386,
0.000364091
])

Var = np.array([
1.38978E-10,
2.75665E-10,
1.94768E-10,
1.70237E-09,
5.77511E-10,
7.44259E-10,
1.61224E-09,
7.51942E-11,
4.98891E-11,
2.41366E-10,
6.90547E-10
])

Std = np.sqrt(Var)

DbT = np.log(2) / Mean

plt.subplot(1,2,1)
plt.errorbar(Label, Mean, yerr = Std, capsize=5, fmt='o', markersize=5, ecolor='blue', markeredgecolor = "blue", color='b')
plt.xlabel("Strain", fontsize = 24)
plt.ylabel("Growth Rate [1/s]", fontsize = 24)

plt.subplot(1,2,2)
plt.scatter(Label, DbT, color='b')
plt.xlabel("Strain", fontsize = 24)
plt.ylabel("Doubling Time [s]", fontsize = 24)

plt.show()




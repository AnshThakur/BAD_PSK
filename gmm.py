
# program to train and save a GMM as a Mat file.

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.io.matlab import savemat

# load data from text file
data1 = np.genfromtxt('./gmm_bird_data',delimiter=',')
print(data1.shape)
print('Done reading data')

gmm1 = GaussianMixture(n_components=64,covariance_type='diag',verbose=1)
print('initialised GMM')

gmm1.fit(data1)

file_name='gmm'

savemat(file_name, {str(file_name+'_params'): dict(weights=gmm1.weights_, means=gmm1.means_, covariances=gmm1.covariances_)})


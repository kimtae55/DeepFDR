import numpy as np
import nibabel as nib
from numba import njit, prange
import sys 
import os
import time
import scipy
from scipy import interpolate
from scipy.stats import t, norm
from collections import namedtuple
from statsmodels.sandbox.stats.multicomp import multipletests
import numba
from numba import cuda, float32, int32, guvectorize, vectorize, config, float64
import time
import torch 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

WikiWelchResult = namedtuple('WikiWelchResult', ('statistic', 'df'))

root = '/Users/taehyo/Dropbox/NYU/Research/Research/Code/data/ADNI/ADNI/'

@njit(parallel=True, nogil=True, cache=True)
def bound(volume):  
    """ 
    Bounding function to bound large arrays and np.memmaps
    volume: A 3D np.array or np.memmap
    """
    count = 0
    threshold = 0

    mins = np.array(volume.shape)
    maxes = np.zeros(3)
    for z in prange(volume.shape[0]):
        for y in range(volume.shape[1]):
            for x in range(volume.shape[2]):
                if volume[z,y,x] != threshold:
                    count += 1
                    if z < mins[0]:
                        mins[0] = z
                    elif z > maxes[0]:
                        maxes[0] = z
                    if y < mins[1]:
                        mins[1] = y
                    elif y > maxes[1]:
                        maxes[1] = y
                    if x < mins[2]:
                        mins[2] = x
                    elif x > maxes[2]:
                        maxes[2] = x

    return mins, maxes, count

def setup():
	with np.printoptions(threshold=np.inf):
		preprocessed_AD = os.path.join(root, 'EMCI')
		preprocessed_CN = os.path.join(root, 'CN')
		if not os.path.exists(preprocessed_CN):
			os.makedirs(preprocessed_CN)

		if not os.path.exists(preprocessed_AD):
			os.makedirs(preprocessed_AD)			

		index_AD, index_CN = get_adni_status()
		roi_img = get_roi_map()

		mins, maxes, count = bound(roi_img)
		maxes = maxes.astype(int)

		np.save(os.path.join(preprocessed_AD, 'EMCI.npy'), np.empty((len(index_AD)-1, maxes[0] + 1 - mins[0], maxes[1] + 1 - mins[1], maxes[2] + 1 - mins[2])))
		np.save(os.path.join(preprocessed_CN, 'CN.npy'), np.empty((len(index_CN)-1, maxes[0] + 1 - mins[0], maxes[1] + 1 - mins[1], maxes[2] + 1 - mins[2])))

def preprocess():
	'''
	all the parameters within memmap is very important - be careful when using it

	AD: 295 images
	CN: 294 images

	size of ROI cube: (94, 119, 98)
	# relevant voxels: 493856
	'''
	start = time.time()

	preprocessed_AD = os.path.join(root, 'EMCI')
	preprocessed_CN = os.path.join(root, 'CN')

	index_AD, index_CN = get_adni_status()

	roi_img = get_roi_map()
	mins, maxes, count = bound(roi_img)
	maxes = maxes.astype(int)
	custom_start = 6

	saveimg = np.memmap(os.path.join(preprocessed_AD, 'EMCI.npy'), dtype='float64', mode='r+', shape=(len(index_AD)-1, maxes[0]-mins[0]+1, maxes[1]-mins[1]+1, maxes[2]-mins[2]+1)) # write to here
	cur = 0
	for index in index_AD:
		if index > custom_start:
			img = np.load(os.path.join(root, 'ADNI_FDGPET_Clinica_pet-volume_1536.npy'), mmap_mode='r')[index]
			preprocessed_img = img.copy()
			for x in range(roi_img.shape[0]):
				for y in range(roi_img.shape[1]):
					for z in range(roi_img.shape[2]):
						if not roi_img[x,y,z]:
							preprocessed_img[x,y,z] = 0	
			preprocessed_img = preprocessed_img[mins[0]:maxes[0]+1, mins[1]:maxes[1]+1, mins[2]:maxes[2]+1]
			saveimg[cur] = preprocessed_img

			print("Time: " + str(cur) + ", " + str(time.time() - start), end='\r')
			cur += 1

	saveimg.flush()

	
	saveimg = np.memmap(os.path.join(preprocessed_CN, 'CN.npy'), dtype='float64', mode='r+', shape=(len(index_CN)-1,maxes[0]-mins[0]+1, maxes[1]-mins[1]+1, maxes[2]-mins[2]+1)) # write to here

	cur = 0
	for index in index_CN:
		if index > custom_start:
			img = np.load(os.path.join(root, 'ADNI_FDGPET_Clinica_pet-volume_1536.npy'), mmap_mode='r')[index]
			preprocessed_img = img.copy()
			for x in range(roi_img.shape[0]):
				for y in range(roi_img.shape[1]):
					for z in range(roi_img.shape[2]):
						if not roi_img[x,y,z]:
							preprocessed_img[x,y,z] = 0				
			preprocessed_img = preprocessed_img[mins[0]:maxes[0]+1, mins[1]:maxes[1]+1, mins[2]:maxes[2]+1]
			saveimg[cur] = preprocessed_img
			print("Time: " + str(cur) + ", " + str(time.time() - start), end='\r')
			cur += 1

	saveimg.flush()

def get_roi_map():
	# get ROI region, set background voxels as 0
	# extract cropped image 
	roi = nib.load(os.path.join(root, 'labels_Neuromorphometrics.nii'))
	roi_img = np.array(roi.dataobj)

	rois = set()
	with open(os.path.join(root, 'spm_templates.man'), 'r') as file:
		for i, line in enumerate(file):
			if i >= 87 and i <= 222:
				roi_label = int(line.split()[1])
				rois.add(roi_label)


	exclude = {4, 11, 44, 45, 63, 64, 46, 51, 52, 61, 62, 69, 49, 50}
	rois = rois - exclude
	for i in range(roi_img.shape[0]):
		for j in range(roi_img.shape[1]):
			for k in range(roi_img.shape[2]):
				if roi_img[i,j,k] in rois:
					roi_img[i,j,k] = 1
				else:
					roi_img[i,j,k] = 0

	mins, maxes, count = bound(roi_img)
	maxes = maxes.astype(int)
	roi_img = roi_img[mins[0]:maxes[0]+1, mins[1]:maxes[1]+1, mins[2]:maxes[2]+1]

	return roi_img

def get_adni_status():
	# get list of EMCI and CN files
	index_AD = []
	index_CN = []
	with open(os.path.join(root, 'ADNI_FDGPET_subjectID_1536_status.txt'), 'r') as file:
		for i, line in enumerate(file):
			if line.split()[1] == 'EMCI':
				index_AD.append(i-1)
			elif line.split()[1] == 'CN':
				index_CN.append(i-1)
	return index_AD, index_CN

def welch_ttest(a, b):
	"""
	Welch's t-test based on wikipedia: https://en.wikipedia.org/wiki/Welch%27s_t-test
	I want to compare the formulas with the one given by scipy, and if they're consistent, I will use the DoF formula to 
	perform the transformation to z-score

	Confirmed that result is same as scipy.stats.ttest_ind(AD_i, CN_i, equal_var=False, alternative='two-sided')
	"""

	n1 = len(a)
	n2 = len(b)

	a_mean = np.mean(a)
	b_mean = np.mean(b)

	v1 = np.var(a, ddof=1)
	v2 = np.var(b, ddof=1)

	vn1 = v1 / n1
	vn2 = v2 / n2

	with np.errstate(divide='ignore', invalid='ignore'):
		df = (vn1 + vn2)**2 / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

	# If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
	# Hence it doesn't matter what df is as long as it's not NaN.
	df = np.where(np.isnan(df), 1, df)

	t = (a_mean - b_mean) / (np.sqrt(vn1 + vn2))

	return WikiWelchResult(t, df)

def test_statistic():
	"""
	Compute t-value using Welch's t-test.
	Then t-value and v are put into the t-distribution’s cdf to get G(t)
	Then inverse cdf of standard Gaussian is used to convert it to z-value.
	We use z-value because we assume the standard Gaussian for the null distribution.

	Use p-value returned from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html for bh and q-value methods
	"""
	roi_map = get_roi_map()
	index_AD, index_CN = get_adni_status()

	AD = np.memmap(os.path.join(root, 'EMCI', 'EMCI.npy'),  dtype='float64', mode='r', shape=(len(index_AD)-1, roi_map.shape[0], roi_map.shape[1], roi_map.shape[2])) 
	CN = np.memmap(os.path.join(root, 'CN', 'CN.npy'),  dtype='float64', mode='r', shape=(len(index_CN)-1, roi_map.shape[0], roi_map.shape[1], roi_map.shape[2])) 

	test_statistic = np.zeros(AD.shape[1:])
	p_value = np.zeros(AD.shape[1:])
	for i in range(AD.shape[1]):
		for j in range(AD.shape[2]):
			for k in range(AD.shape[3]):
				if roi_map[i,j,k]:
					AD_i = AD[:,i,j,k]
					CN_i = CN[:,i,j,k]
					result = welch_ttest(AD_i, CN_i)
					G = t.cdf(x = result.statistic, df = result.df)
					z = norm.ppf(G)
					p_value[i,j,k] = scipy.stats.ttest_ind(AD_i, CN_i, equal_var=False).pvalue
					test_statistic[i,j,k] = z

	np.save(os.path.join(root,'x_3d.npy'), test_statistic)
	np.save(os.path.join(root,'p_value.npy'), p_value)

def test_p_value(p_value, threshold):
	pv = np.ravel(p_value)
	reject = np.zeros(pv.shape)
	for i in range(reject.shape[0]):
		if pv[i] < threshold:
			reject[i] = 1
	reject = reject.reshape(p_value.shape).astype('float')
	return reject

def find_signals_bh(p_value, threshold):
	# calculate p_value from z-score
	#p_value = scipy.stats.norm.sf(np.fabs(x))*2.0 # two-sided tail, calculates 1-cdf
	p_value = np.ravel(p_value)
	reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_value, alpha=threshold, method='fdr_bh')
	reject = reject.reshape(x.shape)
	reject = reject.astype('float')
	return reject

def find_signals_q(pv, threshold):
	"""
	Estimates q-values from p-values
	Args
	=====
	m: number of tests. If not specified m = pv.size
	verbose: print verbose messages? (default False)
	lowmem: use memory-efficient in-place algorithm
	pi0: if None, it's estimated as suggested in Storey and Tibshirani, 2003.
	     For most GWAS this is not necessary, since pi0 is extremely likely to be
	     1
	"""
	m=None
	verbose=False
	lowmem=False
	pi0=None

	#pv = scipy.stats.norm.sf(np.fabs(x))*2.0 # two-sided tail, calculates 1-cdf
	pv = np.ravel(pv)

	assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

	if m is None:
	    m = float(len(pv))
	else:
	    # the user has supplied an m
	    m *= 1.0

	# if the number of hypotheses is small, just set pi0 to 1
	if len(pv) < 100 and pi0 is None:
	    pi0 = 1.0
	elif pi0 is not None:
	    pi0 = pi0
	else:
	    # evaluate pi0 for different lambdas
	    pi0 = []
	    lam = np.arange(0, 0.90, 0.01)
	    counts = np.array([(pv > i).sum() for i in np.arange(0, 0.9, 0.01)])
	    for l in range(len(lam)):
	        pi0.append(counts[l]/(m*(1-lam[l])))

	    pi0 = np.array(pi0)

	    # fit natural cubic spline
	    tck = interpolate.splrep(lam, pi0, k=3)
	    pi0 = interpolate.splev(lam[-1], tck)
	    if verbose:
	        print("qvalues pi0=%.3f, estimated proportion of null features " % pi0)

	    if pi0 > 1:
	        if verbose:
	            print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
	        pi0 = 1.0

	assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

	if lowmem:
	    # low memory version, only uses 1 pv and 1 qv matrices
	    qv = np.zeros((len(pv),))
	    last_pv = pv.argmax()
	    qv[last_pv] = (pi0*pv[last_pv]*m)/float(m)
	    pv[last_pv] = -np.inf
	    prev_qv = last_pv
	    for i in range(int(len(pv))-2, -1, -1):
	        cur_max = pv.argmax()
	        qv_i = (pi0*m*pv[cur_max]/float(i+1))
	        pv[cur_max] = -np.inf
	        qv_i1 = prev_qv
	        qv[cur_max] = min(qv_i, qv_i1)
	        prev_qv = qv[cur_max]

	else:
	    p_ordered = np.argsort(pv)
	    pv = pv[p_ordered]
	    qv = pi0 * m/len(pv) * pv
	    qv[-1] = min(qv[-1], 1.0)

	    for i in range(len(pv)-2, -1, -1):
	        qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

	    # reorder qvalues
	    qv_temp = qv.copy()
	    qv = np.zeros_like(qv)
	    qv[p_ordered] = qv_temp

	# reject null hypothesis using fdr threshold
	reject = np.zeros(qv.shape)
	for i, val in enumerate(qv):
		if val < threshold: # reject
			reject[i] = 1
	reject = reject.reshape(x.shape)
	return reject

def find_signals_lis():
	pass

@vectorize([float32(float32, float32, float32, float32, float32, float32)], target = 'cpu')
def d_ij(x, y, z, i, j, k):
    # create following guvectorize functions:
    # 1. d_ij
    # 2. rho_ij
    # 3. theta_ij
    # 4. theta_sum
    # 5. hj_theta_sum
    # test on gpu, see if it has speed improvement versus prange vs cpu vs parallel vs cuda
	return float32(1.0) / (float32(1.0) + (((z - k) ** 2 + (y - j) ** 2 + (x - i) ** 2)**float32(0.5)))

    # USE THIS FOR THE CODE
    #return float32(1.0) / (float32(1.0) + (((z - k) ** 2 + (y - j) ** 2 + (x - i) ** 2)**float32(0.5)))

def precompute_distance_matrix():
	"""
	We will use np.memmap since the seeks are simple, so its performance will be similar to that of pytables or h5py

	Size of ROI cube is (94, 119, 98)
	What we only use is 493856 voxels. 
	Then for distance matrix, we create a matrix of shape (493856, 493856)
	indexed by each voxel, and it stores distance from voxel i to voxel j

	We only need the upper triangle of this matrix.... but extraneous slicing operation will just slow computation for gibb's sampling
	Start off with (493856,3) matrix - each storing the x,y,z coordinate for each roi voxel 
	-----------------------------------------------------------------------------------------
	Parameters:
	root - root directory of datapath (where it contains all ADNI related files)
	"""
	test_statistic = np.load(os.path.join(root, 'x_3d.npy'))
	roi_map = get_roi_map()
	num_rois = np.count_nonzero(roi_map)

	index = 0
	test_statistic_1d = np.zeros(num_rois)
	xyz = np.zeros((num_rois, 3)).astype(np.float32)
	for i in range(roi_map.shape[0]):
		for j in range(roi_map.shape[1]):
			for k in range(roi_map.shape[2]):
				if roi_map[i,j,k]:
					xyz[index] = np.array([i,j,k]).astype(np.float32)
					test_statistic_1d[index] = test_statistic[i,j,k]
					index += 1

	# now I have the x,y,z coordinate for all roi voxels, and also the corresponding test_statistics for p voxels 
	# save this so that I can use it for mapping later 
	print("num_voxels: ", test_statistic_1d.shape)
	np.save(os.path.join(root, '1d_roi_to_3d_mapping.npy'), xyz)
	np.save(os.path.join(root, 'x_1d.npy'), test_statistic_1d)
	
	# use the below snippet when computing the distance
	'''
	np.save('/Users/taehyo/Dropbox/NYU/Research/Research/Code/data/ADNI/ADNI/distance.npy', np.empty((num_rois, num_rois), dtype=np.float16))

	dist = np.memmap('/Users/taehyo/Dropbox/NYU/Research/Research/Code/data/ADNI/ADNI/distance.npy', dtype='float16', mode='r+', shape=(num_rois,num_rois)) # write to here

	for index in range(num_rois):
	    dist[index] = d_ij(xyz[:,0], xyz[:,1], xyz[:,2], xyz[index,0], xyz[index,1], xyz[index,2]).astype(np.float16)

	dist.flush()
	'''
if __name__ == '__main__':
	start = time.time()

	#setup()
	#preprocess()
	#test_statistic()
	#precompute_distance_matrix()

	
	roi_map = get_roi_map()
	index_AD, index_CN = get_adni_status()
	roi = nib.load('/Users/taehyo/Documents/NYU/Research/DeepFDR/cHMRF/roi.nii')
	roi_img = np.array(roi.dataobj)
	x = np.load('/Users/taehyo/Dropbox/NYU/Research/Research/Code/data/ADNI/ADNI/p_value.npy')

	bh_signals = find_signals_bh(x, 1e-3)
	q_signals = find_signals_q(x, 1e-3)
	p_signals = test_p_value(x, 1e-3)

	irrelevant = roi_img.shape[0] * roi_img.shape[1] * roi_img.shape[2] - np.count_nonzero(roi_map)
	print(np.count_nonzero(bh_signals)-irrelevant, np.count_nonzero(q_signals)-irrelevant, np.count_nonzero(p_signals)-irrelevant)
	cropped = nib.Nifti1Image(bh_signals*roi_img, np.eye(4))
	cropped.header.get_xyzt_units()
	cropped.to_filename('./bh.nii')

	cropped = nib.Nifti1Image(q_signals*roi_img, np.eye(4))
	cropped.header.get_xyzt_units()
	cropped.to_filename('./q.nii')	

	cropped = nib.Nifti1Image(p_signals*roi_img, np.eye(4))
	cropped.header.get_xyzt_units()
	cropped.to_filename('./p.nii')	
	

	'''
	roi_map = get_roi_map()
	index_AD, index_CN = get_adni_status()

	cropped = np.memmap(os.path.join(root, 'EMCI', 'EMCI.npy'),  dtype='float64', mode='r', shape=(len(index_AD)-1, roi_map.shape[0], roi_map.shape[1], roi_map.shape[2]))[5] 
	cropped = nib.Nifti1Image(cropped, np.eye(4))
	cropped.header.get_xyzt_units()
	cropped.to_filename('./roi_CN.nii')
	'''


	print("time elapsed: ", time.time() - start)










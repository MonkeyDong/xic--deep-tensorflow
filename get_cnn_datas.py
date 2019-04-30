import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os

#labels -> one_hot
def one_hot(y):
	lb = LabelBinarizer()
	lb.fit(y)
	yy = lb.transform(y)
	return yy

#Z-score标准化
def z_score(x):
	#x = np.log(x)
	x = (x - np.average(x))/np.std(x)
	return x

def adds(path,num):
	files = os.listdir(path)
	nn = len(files)
	Xt = np.zeros((nn, 256, 32))
	for m,n in enumerate(files):
		ms = np.load(path+"/"+n)
		xs = []
		ss = []
		for j in range(32):
			ss = ms[j*256:(j+1)*256]
			xs.append(ss)
		xs = z_score(xs)
		m_s = np.array(xs).transpose(1,0)
		Xt[m,:,:] = m_s

	labels = [num]*nn
	X = Xt.tolist()
	return X,labels
#################################
#训练集合并
def get_train_datas(ks):
	X2,labels_t2 = adds('datas_mz/MMt',0)
	X3,labels_t3 = adds('datas_mz/AAt',0)
	X4,labels_t4 = adds('datas_mz/CCt',1)
	X5,labels_t5 = adds('datas_mz/PPt',1)
	X7,labels_s2 = adds('datas_mz/MMs',0)
	X8,labels_s3 = adds('datas_mz/AAs',0)
	X9,labels_s4 = adds('datas_mz/CCs',1)
	X10,labels_s5 = adds('datas_mz/PPs',1)

	Xtt = [j for j in X2]+[k for k in X3]+[l for l in X4]+[g for g in X5]
	Xss = [j for j in X7]+[k for k in X8]+[l for l in X9]+[g for g in X10]

	X_train = np.array(Xtt)
	X_test = np.array(Xss)

	labels_train = labels_t2 + labels_t3 + labels_t4 + labels_t5
	labels_test = labels_s2 + labels_s3 + labels_s4 + labels_s5

	X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, stratify = labels_train, test_size=0.05,random_state = 123)

	y_tr = one_hot(lab_tr)
	y_vld = one_hot(lab_vld)
	y_test = one_hot(labels_test)

	np.save("save_npy/x_train.npy",X_tr)
	np.save("save_npy/x_ver.npy",X_vld)
	np.save("save_npy/X_test.npy",X_test)
	np.save("save_npy/y_train.npy",y_tr)
	np.save("save_npy/y_ver.npy",y_vld)
	np.save("save_npy/y_test.npy",y_test)
	print("cross-",ks,"训练数据处理完成")




import numpy as np
import os

paths = ['datas/AA','datas/CC','datas/MM','datas/PP']

def get_rand_vecs():
	lis = np.load("datas/vecs.npy")
	n = 4096
	indexs = np.random.choice(lis.shape[0],n,replace=False)
	lists = lis[indexs]
	np.save("save_npy/vecs_key.npy",lists)
	return lists

def re_sampling(path):
	ffile = os.listdir("datas_mz/"+path[-2:]+"t")
	nn = len(ffile)
	diff = 500 - nn
	step = int(diff/nn) + 1
	oo = diff%nn
	while(step > 0):
		if step == 1:
			times = oo
		else:
			times = nn
		for i in range(times):
			fs = np.load("datas_mz/"+path[-2:]+"t/"+ffile[i])
			ti = ffile[i][:-4]
			np.save("datas_mz/"+path[-2:]+"t/"+ti+"cp"+str(step)+".npy",fs)
			print("datas_mz/"+path[-2:]+"t/"+ti+"cp"+str(step)+".npy")
		step -= 1

def get_vecs(path,lists,low,high):
	files = os.listdir(path)
	for l in files:
		res = []
		arr = []
		f = np.load(path+'/'+l)
		x1 = f[:,0]
		x2 = f[:,1]
		me1,me2 = np.percentile(x1, [low, high])
		for i in range(len(x1)):
			if x1[i] > me1 and x1[i] < me2:
				arr.append(i)
		ff = f[arr]
		for j in lists:
			ss = ff[np.argwhere(ff[:,1]==j)][:,0]
			if ss.tolist() == []:
				ss = [0]
			res.append(np.mean(ss))
			res.append(np.std(ss))
		np.save("datas_mz/"+l[:3]+"/"+l,res)
		print("datas_mz/"+l[:3]+"/"+l)

def get_vec_mz(ks,key,l,h):
	for path in paths:
		get_vecs(path,key,l,h)
		re_sampling(path)
	print("cross-",ks,"特征向量提取完成")


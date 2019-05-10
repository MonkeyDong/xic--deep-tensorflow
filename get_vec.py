import numpy as np
import os

paths = [('AA',196),('CC',107),('MM',262),('PP',141)]
l = 20
h = 90


def adds(ll):
	vecs = []
	for i in ll:
		if i not in vecs:
			vecs.append(i)
	return vecs

def plots(path,low,high):
	files = os.listdir('datas/'+path[0])
	tt = 0
	arr = []
	for f in files:
		vv = []
		f = np.load('datas/'+path[0]+'/'+f)
		x1 = f[:,0]
		x2 = f[:,1]
		me1,me2 = np.percentile(x1, [low, high])
		for i in range(len(x1)):
			if x1[i] > me1 and x1[i] < me2:
				vv.append(x2[i])
		arr.append(list(set(vv)))
	dic = {}
	for k in arr:
		for j in k:
			if j in dic.keys():
				dic[j] += 1
			else:
				dic[j] = 1
	ll = []
	for i in dic.keys():
		if dic[i]>path[1]:
			ll.append(i)
	np.save("datas/vec_npy/"+path[0][0]+"_vec.npy",ll)
	print(path,len(ll))

def get_vecs_key(paths,l,h):		
	for path in paths:
		plots(path,l,h)

	m = np.load('datas/vec_npy/M_vec.npy')
	a = np.load('datas/vec_npy/A_vec.npy')
	c = np.load('datas/vec_npy/C_vec.npy')
	p = np.load('datas/vec_npy/P_vec.npy')

	v1 = adds(m)
	v2 = adds(a)
	v3 = adds(c)
	v4 = adds(p)
	vv = list(set(v1+v2+v3+v4))

	np.save('datas/vecs.npy',vv)
	print("The vecs-key is\n",vv)
	print("Len(vecs-key) = ",len(vv))



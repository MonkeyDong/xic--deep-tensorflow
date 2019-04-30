from cross_mean import split_cross
from load_datas import load
from get_vec import get_vecs_key
from choose import get_rand_vecs,get_vec_mz
from get_cnn_datas import get_train_datas
from metrics_result import res_metrics
import shutil
import os

paths = [('AA',360),('CC',189),('MM',472),('PP',284)]
l = 20
h = 90


for i in range(10):
	split_cross(i)
	load(i)
	if i == 0:
		get_vecs_key(paths,l,h)
		key = get_rand_vecs()
	get_vec_mz(i,key,l,h)
	get_train_datas(i)
	os.system('python3 nx1_cnn.py '+str(i+1))
	res_metrics(i)

	pp1 = ['AA','CC','MM','PP','vec_npy']
	pp2 = ['AAs','AAt','CCs','CCt','MMs','MMt','PPs','PPt']
	shutil.rmtree('datas')
	shutil.rmtree('datas_mz')
	os.mkdir('datas')
	os.mkdir('datas_mz')
	for i in pp1:	
		os.mkdir('datas/'+i)
	for j in pp2:
		os.mkdir('datas_mz/'+j)




	

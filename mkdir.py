import os
import shutil

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

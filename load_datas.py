import numpy as np
import xlrd
import os
import re

workbook1 = xlrd.open_workbook('raw_datas/TPD_training_samples_lable V20181016.xlsx')
booksheet1 = workbook1.sheet_by_index(0)
dics1 = {}  #样本名：ID号
rows = booksheet1.nrows
for i in range(rows):
	key_ = booksheet1.cell_value(i,3)
	cell_ = booksheet1.cell_value(i,2)
	dics1[key_] = cell_

workbook2 = xlrd.open_workbook('raw_datas/TPD_IPX0001444002.xlsx')
booksheet2 = workbook2.sheet_by_index(0)
dics2 = {}
rows = booksheet2.nrows
for i in range(rows):
	key_ = booksheet2.cell_value(i,2)
	cell_ = booksheet2.cell_value(i,1)
	dics2[key_] = cell_

def load_data(file_path):
	ll = np.array([9,12])
	bars = []
	f = open(file_path,'r')
	data = f.readlines()
	for ii in data[1:]:
		cc = ii.split('\n')[0]
		cc = cc.split('\t')
		cc = (np.array(list(map(float,cc))))[ll]
		#print(cc)
		cc[1] = cc[1]+0.0001
		cc[1] = float(re.findall(r"\d{1,}?\.\d{2}", str(cc[1]))[0])
		bars.append(cc)
	bars = sorted(bars, key=lambda arr: arr[1])
	return bars

def loads(files,f_name,ff):
	mm = len(files)
	tt = 0
	for k in files:
		print(k)
		bbar = load_data(os.path.join(ff,k))
		np.save('datas/'+f_name[:-1]+'/'+f_name+ff[13]+'%d.npy' % (tt),bbar)
		tt = tt+1

def load_file(f,st,ffname):
	if ffname[13] == '1':	
		dics = dics1
	if ffname[13] == '2':
		dics = dics2
	file11 = []
	file11 = os.listdir(ffname)
	fs = np.load(f)
	ffs = fs.tolist()

	arr = []
	for i in range(len(ffs)):
		if ffs[i] not in file11:
			arr.append(ffs[i])
	for i in arr:
		ffs.remove(i)

	AA = []
	MM = []
	CC = []
	PP = []
	for i in ffs:
		if not (i[:-7] in dics.keys()):
			continue 
		if dics[i[:-7]] == "":
			continue
		if dics[i[:-7]][:2] == "po":
			continue
		if dics[i[:-7]][:2] == "Mo":
			continue
		if dics[i[:-7]][0] == "N":
			continue
		if dics[i[:-7]][0] == "A":
			AA.append(i)
		if dics[i[:-7]][0] == "M":
			MM.append(i)
		if dics[i[:-7]][0] == "C":
			CC.append(i)
		if dics[i[:-7]][0] == "P":
			PP.append(i)

	loads(MM,'MM'+st,ffname)#根据对应的列表读文件取数据保存到相应的文件夹中
	loads(AA,'AA'+st,ffname)
	loads(PP,'PP'+st,ffname)
	loads(CC,'CC'+st,ffname)

def load(ks):
	load_file("save_npy/file1.npy",'s','raw_datas/TPD1XICS')
	load_file("save_npy/file2.npy",'t','raw_datas/TPD1XICS')
	load_file("save_npy/file3.npy",'s','raw_datas/TPD2XICS')
	load_file("save_npy/file4.npy",'t','raw_datas/TPD2XICS')
	print("cross-",ks," 加载数据")



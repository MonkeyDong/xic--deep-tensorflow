import numpy as np
import xlrd
import os

workbook1 = xlrd.open_workbook('raw_datas/TPD_training_samples_lable V20181016.xlsx')
booksheet1 = workbook1.sheet_by_index(0)
workbook2 = xlrd.open_workbook('raw_datas/TPD_IPX0001444002.xlsx')
booksheet2 = workbook2.sheet_by_index(0) 

#用字典的形式保存excel表格中的 ID：样本名
dics1 = {}
rows = booksheet1.nrows
for i in range(1,rows):
	key_ = booksheet1.cell_value(i,3)
	cell_ = booksheet1.cell_value(i,2)
	if cell_[:2] == "Mo":
		continue
	if cell_[:2] == "po":
		continue
	if cell_[:1] == "N":
		continue
	if cell_ == "":
		continue
	dics1[key_] = cell_ 
#对应的文件去重
dics1 = {value:key for key,value in dics1.items()}

#去除表格与文件无法对应的样本
ff1 = os.listdir("raw_datas/TPD1XICS")
kks = []
for i in dics1.values():
	fs = i+"out.txt"
	if fs not in ff1:
		kk = list(dics1.keys())[list(dics1.values()).index(i)]
		kks.append(kk)
for j in kks:
	del dics1[j]
######
dics2 = {}
rows = booksheet2.nrows
for i in range(1,rows):
	key_ = booksheet2.cell_value(i,2)
	if key_ == "":
		continue
	cell_ = booksheet2.cell_value(i,1)
	dics2[key_] = cell_
dics2 = {value:key for key,value in dics2.items()}

ff2 = os.listdir("raw_datas/TPD2XICS")
kks = []
for i in dics2.values():
	fs = i+"out.txt"
	if fs not in ff2:
		kk = list(dics2.keys())[list(dics2.values()).index(i)]
		kks.append(kk)
for j in kks:
	del dics2[j]

#以字典的形式保存 病例号：[该病例号的样本]
data1 = {}
for d in dics1.keys():
	if d[:-1] in data1.keys():
		data1[d[:-1]].append(d)
	else:
		data1[d[:-1]] = []
		data1[d[:-1]].append(d) #357个病例

data2 = {}
for d in dics2.keys():
	if d[:-1] in data2.keys():
		data2[d[:-1]].append(d)
	else:
		data2[d[:-1]] = []
		data2[d[:-1]].append(d) #180个病例

'''
#求两批样本中的样本数量
nn = 0
for i in data1.keys():
     nn += len(data1[i])#905

mm = 0
for i in data2.keys():
     mm += len(data2[i])#536
'''
def class_key(data,key):
	dc = []
	for i in data:
		if i[0] == key:
			dc.append(i)
	return dc

def splits(datas,n):
	das = []
	da2 = np.array(datas)
	for i in range(9):
		index_1=np.random.choice(da2.shape[0],n,replace=False)
		da1=da2[index_1].tolist()
		index_2=np.arange(da2.shape[0])
		index_2=np.delete(index_2,index_1)
		da2=da2[index_2]
		das.append(da1)
	da2 = da2.tolist()
	if len(da2) <= n:
		das.append(da2)
	if len(da2) > n:
		das.append(da2[:n])
		del da2[:n]
		for i,j in enumerate(da2):
			das[i].append(j)
	return das

def adds(dd,ti):
	if ti == 1:
		data = data1
		dics = dics1
	if ti == 2:
		data = data2
		dics = dics2
	ddd = []
	for n in data.keys():
		if n in dd:
			for m in data[n]:
				ddd.append(m)
	f = []
	for i in ddd:
		if dics[i] == "":
			continue
		else:
			f.append(dics[i]+"out.txt")
	return f

def split_class(dm,ks,x):
	das1 = splits(dm,int(len(dm)/10))
	dd1 = das1[ks]
	dds2 = das1[:ks]+das1[ks+1:]
	dd2 = []	
	for i in dds2:
		dd2 = dd2 + i
	f1 = adds(dd1,x)
	f2 = adds(dd2,x)
	return f1,f2

def split_cross(ks):
	dm1 = class_key(data1,"M")
	dm2 = class_key(data2,"M")
	da1 = class_key(data1,"A")
	da2 = class_key(data2,"A")
	dc1 = class_key(data1,"C")
	dc2 = class_key(data2,"C")
	dp1 = class_key(data1,"P")
	dp2 = class_key(data2,"P")

	fm1,fm1s = split_class(dm1,ks,1)
	fm2,fm2s = split_class(dm2,ks,2) 
	fa1,fa1s = split_class(da1,ks,1)
	fa2,fa2s = split_class(da2,ks,2)
	fc1,fc1s = split_class(dc1,ks,1)
	fc2,fc2s = split_class(dc2,ks,2)
	fp1,fp1s = split_class(dp1,ks,1)
	fp2,fp2s = split_class(dp2,ks,2)

	f2 = fm1s + fa1s + fc1s + fp1s #808
	f1 = fm1 + fa1 + fc1 + fp1 #97
	f4 = fm2s + fa2s + fc2s + fp2s #479
	f3 = fm2 + fa2 + fc2 + fp2 #57

	#少了几个数据
	np.save("save_npy/file1.npy",f1)
	np.save("save_npy/file2.npy",f2)
	np.save("save_npy/file3.npy",f3)
	np.save("save_npy/file4.npy",f4)
	print("cross-",ks,"(样本划分):",len(f1),len(f2),len(f3),len(f4))


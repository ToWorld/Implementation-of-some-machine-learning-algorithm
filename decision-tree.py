from math import log 

# 计算信息熵
def entropy(sqmple):
	log2 = lambda x:log(x)/log(2)

	results = {}
	for row in sample:
		r = row[len(row)-1]
		results[r] = results.get(r, 0) + 1

	ent = 0.0	# entropy 
	for r in results.keys():
		p = float(results[r])/len(sample)
		ent -= p*log2(p)	
	return ent

# 获取数据	
def fetch_subdataset(sample, k, v):
	return [d[:k] + d[k+1:] for d in sample if d[k] == v]		

# 样本只包含一个字段，将类别标记为样本数最多的类
def MaxFeature(classList):
	classCount = {}
	for cla in classList:
		classCount[cla] = classCount.get(cla, 0) + 1
	sortedClassCount = sorted(classCount.items(), key=lambda d: d[1], reverse=True)
	return sortedClassCount[0][0]		

# 确定当前最优特征
def DecisionFeature(sample):
	ent, feature = 100000000, -1
	# 计算每一个字段的信息熵
	for i in range(len(sample[0])-1):
		feat_list = [e[i] for e in sample]	# 特定属性列的所有样本取值
		unq_feat_list = set(feat_list)	# 去重
		ent_t = 0.0
		# 计算没有当前属性各个其值的熵并相加得到这个属性的信息熵
		for f in unq_feat_list:	
			# sample : 样本
			# i : 特定属性
			# f : 特定属性的特定取值
			# return : 
			sub_data = fetch_subdataset(sample, i, f)
			ent_t += entropy(sub_data)*len(sub_data)/len(sample)

		if ent_t < ent:
			ent, feature = ent_t, i 
	return feature 
	
# sample : 样本
# datalabel : 样本属性字段
def buildTree(sample, datalabel):
	cla = [c[-1] for c in sample]	# 样本类别(样本类别一般是在最后一列)

	# 样本全属于同一类别
	if len(cla) == cla.count(cla[0]):
		return cla[0]

	# 样本只包含一个字段，将类别标记为样本数最多的类	
	if len(sample[0]) == 1:
		return MaxFeature(sample)

	# 其他情况 [找出当前树的根结点]	
	feature = DecisionFeature(sample)	# 选择当前最优划分属性(即通过计算熵，计算熵最大值)
	featureLabel = datalabel[feature]	# 
	decisionTree = {featureLabel:{}}	# 开始生成子树					
	del(datalabel[feature])	# 去掉已经使用过的属性

	featValue = [d[feature] for d in sample]	# 获得所有样本的特定属性的取值
	UniqueFeatValue = set(featValue)	# 特定属性取值可能并去重
	for value in UniqueFeatValue:
		subLabel = datalabel[:]	
		decisionTree[featureLabel][value] = buildTree(fetch_subdataset(sample,feature, value), subLabel)	# 递归生成决策树

	return decisionTree

# 使用生成的决策树
def classify(decisionTree, featLabels, test):
	label = decisionTree.key()[0]		
	next_dict = decisionTree[label]
	feat_index = featLabels.index(label)
	for key in next_dict.keys():
		if test[feat_index] == key:
			if type(next_dict[key]).__name__=='dict':
				c_label = classify(next_dict[key], featLabels, test)
			else:
				c_label = next_dict[key]
	return c_label				
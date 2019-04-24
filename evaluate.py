import numpy as np
import codecs, json
import pandas as pd


obj_text = codecs.open('assets/result.json', 'r', encoding='utf-8').read()
obj_np = np.array(json.loads(obj_text))
res = pd.DataFrame(obj_np, columns=['original','predict'])[250:]


thresh_ = res['predict'].value_counts().median()

for i in range(len(res['predict'].value_counts())):
	if res['predict'].value_counts().iloc[i] <= thresh_:
		thresh_ = i
		break


res = res.loc[res['predict'].isin(res['predict'].value_counts().index[thresh_:])].copy()

print("Accuracy : {0:.2f}".format(res.loc[res['original']*res['predict']>0].shape[0]*100/res.shape[0]))
print(res.shape[0])
import pandas as pd
from pandas import json_normalize
import json
import seaborn as sn
from elasticsearch import Elasticsearch
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from gurobipy import Model, GRB, quicksum

es = Elasticsearch([{'host':'127.0.0.1', 'port': 9200}])
es.indices.create(index='kdapom_orders', ignore=400)

#ingest_csv_file_into_elastic_index('sales.csv',es,'kdapom_orders',buffer_size=5000)
margins_df = pd.read_csv(r'margins.csv',encoding='latin1')
dimensions_df = pd.read_csv(r'dimensions.csv',encoding='latin1')
dimensions_df['volume'] = dimensions_df['length'] * dimensions_df['width'] * dimensions_df['height']

no_of_days = 730
no_of_products = len(dimensions_df.axes[0])
df1 = pd.DataFrame()
df = pd.DataFrame()
df['product_id'] = [0] * no_of_products
df['average_order_per_day'] = [0] * no_of_products
df['std_order_per_day'] = [0] * no_of_products

for a in range(1263):
    b = a+1
    search_body = {
        'size': 0,
        'query': {
            'bool':{
                'must':{
                    'term':{
                        "product_id": b
                    }
                }
            }
        },
        "aggs": {
            "days": {
                "terms": {
                    "field" : "day",
                    "size" : 10000
                }
            }
        }
    }
    result = es.search(index="kdapom_orders", body=search_body)
    dfProducts = json_normalize(result['aggregations']['days']['buckets'])
    dfProducts.rename(columns={'key': 'day'}, inplace=True)
    dfProducts.rename(columns={'doc_count': 'number_of_orders'}, inplace=True)
    dfProducts = dfProducts.sort_values(by = ['day'])
    dfProducts = dfProducts.reset_index(drop=True)

    c=0
    if len(dfProducts.axes[0]) < no_of_days:
        gap_days = no_of_days - len(dfProducts.axes[0])
        day = [731] * gap_days
        null_order = [0] * gap_days
        df_temp = pd.DataFrame({'day': day,'number_of_orders': null_order})
        dfProducts = dfProducts.append(df_temp, ignore_index=True)
        if dfProducts.day[0] != 1:
            dfProducts.day[len(dfProducts.axes[0]) - gap_days] = 1
            c += 1
        dfProducts = dfProducts.sort_values(by=['day'])
        dfProducts = dfProducts.reset_index(drop=True)

        for i in range(len(dfProducts.axes[0])-gap_days):
            if (dfProducts.day[i+1] - dfProducts.day[i]) ==1 :
             x=1
            else:
                for j in range(dfProducts.day[i+1] - dfProducts.day[i]-1):
                    dfProducts.day[len(dfProducts.axes[0])-gap_days+c] = dfProducts.day[i] + 1+j
                    c+=1
    dfProducts = dfProducts.sort_values(by = ['day'])
    dfProducts = dfProducts.reset_index(drop=True)

    df1['Product '+str(a+1)] = dfProducts['number_of_orders']

    df.product_id[a] = b
    df.average_order_per_day[a] = dfProducts.mean()[1]
    df.std_order_per_day[a] = dfProducts.std()[1]
#df = df.sort_values(by = ['average_order_per_day'], ascending=False)
#df = df.reset_index(drop=True)
#plt.errorbar(df.index,df.average_order_per_day, xerr=None, yerr=df.std_order_per_day,fmt='o',ecolor='green')
#plt.title('Errorbar')

df['margins'] = margins_df['margin']
df['daily_profit'] = df['average_order_per_day'] * df['margins']
df['volume'] = dimensions_df['volume']
df = df.sort_values(by = ['product_id'])
df = df.reset_index(drop=True)
df.to_csv('product_with_avg_and_margins_volume.csv')

fig, axes = plt.subplots(1, 2)
df.hist('daily_profit', ax=axes[0], bins = 20)
df.hist('volume', ax=axes[1], bins = 20)

a = df['daily_profit'].describe(percentiles = [0.5, 0.8])

dfA = df[df['daily_profit']>a[5]]
dfA = dfA.sort_values(by=['daily_profit'], ascending=False)
dfA = dfA.reset_index(drop=True)
dfB = df[df['daily_profit']>a[4]]
dfB = dfB[dfB['daily_profit']<=a[5]]
dfB = dfB.sort_values(by=['daily_profit'], ascending=False)
dfB = dfB.reset_index(drop=True)
dfC = df[df['daily_profit']<=a[4]]
dfC = dfC.sort_values(by=['daily_profit'], ascending=False)
dfC = dfC.reset_index(drop=True)
clas = ['A','B','C']
sum = [dfA['daily_profit'].sum(),dfB['daily_profit'].sum(),dfC['daily_profit'].sum()]
num_products = [len(dfA.axes[0]),len(dfB.axes[0]),len(dfC.axes[0])]

fig, axes = plt.subplots(1)
plt.subplot(211)
plt.bar(clas, num_products, color='green', width=0.4)
plt.xlabel("Product classes")
plt.ylabel("Number of Products")
plt.subplot(212)
plt.bar(clas, sum,color='blue', width=0.4)
plt.xlabel("Product classes")
plt.ylabel("Total Average Daily profits")
fig, axes = plt.subplots(1)
plt.subplot(311)
plt.bar(dfA.index, dfA.daily_profit, color='red')
plt.subplot(312)
plt.bar(dfB.index, dfB.daily_profit, color='green')
plt.subplot(313)
plt.bar(dfC.index, dfC.daily_profit, color='blue')
plt.ylabel("Average Daily profits")

std_box_volume = 40*40*20*0.9
z_90 = 1.645
z_95 = 1.96
z_99 = 2.576
dfA['average_weekly_demand'] = dfA['average_order_per_day'] * 7
dfA['std_weekly_demand'] = dfA['std_order_per_day'] * math.sqrt(7)
dfA['base_stock'] = dfA['average_weekly_demand'] + dfA['std_weekly_demand'] * z_99
dfA['base_stock_volume'] = dfA['base_stock'] * dfA['volume']
dfA['box_required'] = dfA['base_stock_volume'] / std_box_volume
dfA['daily_profit_loss'] = dfA['daily_profit'] * 0.2
dfA['ratio'] = dfA['daily_profit_loss'] / dfA['box_required']
dfA = dfA.sort_values(by=['product_id'], ascending=False)
dfA = dfA.reset_index(drop=True)
dfA.to_csv('Class_A_with_weekly-demands.csv')

dfB['average_weekly_demand'] = dfB['average_order_per_day'] * 7
dfB['std_weekly_demand'] = dfB['std_order_per_day'] * math.sqrt(7)
dfB['base_stock'] = dfB['average_weekly_demand'] + dfB['std_weekly_demand'] * z_95
dfB['base_stock_volume'] = dfB['base_stock'] * dfB['volume']
dfB['box_required'] = dfB['base_stock_volume'] / std_box_volume
dfB['daily_profit_loss'] = dfB['daily_profit'] * 0.3
dfB['ratio'] = dfB['daily_profit_loss'] / dfB['box_required']
dfB = dfB.sort_values(by=['product_id'], ascending=False)
dfB = dfB.reset_index(drop=True)
dfB.to_csv('Class_B_with_weekly-demands.csv')

dfC['average_weekly_demand'] = dfC['average_order_per_day'] * 7
dfC['std_weekly_demand'] = dfC['std_order_per_day'] * math.sqrt(7)
dfC['base_stock'] = dfC['average_weekly_demand'] + dfC['std_weekly_demand'] * z_90
dfC['base_stock_volume'] = dfC['base_stock'] * dfC['volume']
dfC['box_required'] = dfC['base_stock_volume'] / std_box_volume
dfC['daily_profit_loss'] = dfC['daily_profit'] * 0.5
dfC['ratio'] = dfC['daily_profit_loss'] / dfC['box_required']
dfC = dfC.sort_values(by=['product_id'], ascending=False)
dfC = dfC.reset_index(drop=True)
dfC.to_csv('Class_C_with_weekly-demands.csv')

fig, axes = plt.subplots(1, 3)
dfA.hist('box_required', ax=axes[0], bins = 10, color='green')
plt.suptitle('For Class A, B & C respectively')
dfB.hist('box_required', ax=axes[1], bins = 10, color='red')
dfC.hist('box_required', ax=axes[2], bins = 10, color='blue')

fig, axes = plt.subplots(1)
df_corr = df1.corr(method='pearson')
df_corr.to_csv('df_corr.csv')
hm = sn.heatmap(df_corr, cmap="PiYG")
plt.show()

df_couple = pd.DataFrame(columns=['first_prod','second_prod','corr_value','first_prod_class','second_prod_class'])
df_temp2 = pd.DataFrame([[0,0,0]],columns=['first_prod','second_prod','corr_value'])
count=0
for i in range(1263):
    for j in range(i+1,1263):
        if df_corr.iloc[i,j] >= 0.6 and i != j:
            df_couple = df_couple.append(df_temp2,ignore_index=True)
            df_couple.iloc[count,0]= i+1
            df_couple.iloc[count,1] = j+1
            df_couple.iloc[count,2] = df_corr.iloc[i,j]
            count += 1

df2 = np.unique(df_couple[['first_prod', 'second_prod']].values)
df_uniq = pd.DataFrame(columns=['prod_list', 'clas'])
df_uniq['prod_list'] = df2
for i in range(len(df_uniq)):
    cnt = 0
    for j in range(len(dfA)):
        if df_uniq.iloc[i,0] == dfA.product_id[j]:
            df_uniq.iloc[i,1] = 3
            cnt +=1
            break
    if cnt == 0:
        for k in range(len(dfB)):
            if df_uniq.iloc[i,0] == dfB.product_id[k]:
                df_uniq.iloc[i,1] = 2
                cnt += 1
                break
    if cnt == 0:
        df_uniq.iloc[i,1] = 1

for i in range(len(df_uniq)):
    for j in range(len(df_couple.axes[0])):
        if df_uniq.prod_list[i]==df_couple.first_prod[j]:
            df_couple.iloc[j,3] = df_uniq.clas[i]
for i in range(len(df_uniq)):
    for j in range(len(df_couple.axes[0])):
        if df_uniq.prod_list[i] == df_couple.second_prod[j]:
            df_couple.iloc[j, 4] = df_uniq.clas[i]
df_couple.to_csv('prod_couple_with_class.csv')

for i in range(len(df_couple.axes[0])):
    if df_couple.first_prod_class[i] == df_couple.second_prod_class[i]:
        df_couple = df_couple.drop(i)

df_couple = df_couple.reset_index(drop=True)
df_couple.to_csv('prod_couple_with_class_2.csv')

df_new = dfA
df_new = df_new.append(dfB,ignore_index=True)
df_new = df_new.append(dfC,ignore_index=True)
df_new['box_required'] = df_new['box_required'].apply(np.ceil)
df_new = df_new.sort_values(by=['daily_profit_loss'], ascending=False)
df_new = df_new.reset_index(drop=True)
df_new.to_csv('final_list.csv')

df_new2 = df_new.sort_values(by=['product_id'])
df_new2 = df_new2.reset_index(drop=True)

df_first_way= pd.DataFrame(columns=['product_id','daily_profit','box_required'])
df_temp2 = pd.DataFrame([[0,0,0]],columns=['product_id','daily_profit','box_required'])
no_of_total_boxes = 0
y =0
for i in range(len(df_new.axes[0])):
    if (no_of_total_boxes + df_new.box_required[i]) <= 960:
        no_of_total_boxes = no_of_total_boxes + df_new.box_required[i]
        if df_new2.product_id[df_new.product_id[i]-1] == df_new.product_id[i] and df_new2.daily_profit_loss[df_new.product_id[i]-1] !=0 :
            df_first_way = df_first_way.append(df_temp2, ignore_index=True)
            df_first_way.product_id[y] = df_new.product_id[i]
            df_first_way.daily_profit[y] = df_new.daily_profit[i]
            df_first_way.box_required[y] = df_new.box_required[i]
            df_new2.daily_profit_loss[df_new.product_id[i]-1] = 0
        y += 1
    for j in range(len(df_couple.axes[0])):
        if df_new.product_id[i] == df_couple.first_prod[j]:
            if df_new2.product_id[df_couple.second_prod[j] - 1] == df_couple.second_prod[j] and df_new2.daily_profit_loss[df_couple.second_prod[j] - 1] != 0:
                if (no_of_total_boxes + df_new2.box_required[df_couple.second_prod[j] - 1]) <= 960:
                    df_first_way = df_first_way.append(df_temp2, ignore_index=True)
                    df_first_way.product_id[y] = df_new2.product_id[df_couple.second_prod[j] - 1]
                    df_first_way.daily_profit[y] = df_new2.daily_profit[df_couple.second_prod[j] - 1]
                    df_first_way.box_required[y] = df_new2.box_required[df_couple.second_prod[j] - 1]
                    df_new2.daily_profit_loss[df_couple.second_prod[j] - 1] = 0
                    no_of_total_boxes = no_of_total_boxes + df_new2.box_required[df_couple.second_prod[j] - 1]
                    y += 1
        if df_new.product_id[i] == df_couple.second_prod[j]:
            if df_new2.product_id[df_couple.first_prod[j] - 1] == df_couple.first_prod[j] and df_new2.daily_profit_loss[df_couple.first_prod[j] - 1] != 0:
                if (no_of_total_boxes + df_new2.box_required[df_couple.first_prod[j] - 1]) <= 960:
                    df_first_way = df_first_way.append(df_temp2, ignore_index=True)
                    df_first_way.product_id[y] = df_new2.product_id[df_couple.first_prod[j] - 1]
                    df_first_way.daily_profit[y] = df_new2.daily_profit[df_couple.first_prod[j] - 1]
                    df_first_way.box_required[y] = df_new2.box_required[df_couple.first_prod[j] - 1]
                    df_new2.daily_profit_loss[df_couple.first_prod[j] - 1] = 0
                    no_of_total_boxes = no_of_total_boxes + df_new2.box_required[df_couple.first_prod[j] - 1]
                    y += 1

df_first_way.to_csv('Warehouse_products_first_way.csv')

df_new2 = df_new2.sort_values(by=['product_id'])
df_new2 = df_new2.reset_index(drop=True)
df_new2.to_csv('final_list_for_loss.csv')

df_new = df_new.sort_values(by=['ratio'], ascending=False)
df_new = df_new.reset_index(drop=True)
df_new.to_csv('final_list_2nd_way.csv')

df_new2 = df_new.sort_values(by=['product_id'])
df_new2 = df_new2.reset_index(drop=True)
df_new2.to_csv('final_list_sortedby_prod_ID.csv')

df_second_way= pd.DataFrame(columns=['product_id','daily_profit','box_required'])
df_temp2 = pd.DataFrame([[0,0,0]],columns=['product_id','daily_profit','box_required'])
no_of_total_boxes = 0
y =0
for i in range(len(df_new.axes[0])):
    if (no_of_total_boxes + df_new.box_required[i]) <= 960:
        no_of_total_boxes = no_of_total_boxes + df_new.box_required[i]
        if df_new2.product_id[df_new.product_id[i]-1] == df_new.product_id[i]:
            df_second_way = df_second_way.append(df_temp2, ignore_index=True)
            df_second_way.product_id[y] = df_new.product_id[i]
            df_second_way.daily_profit[y] = df_new.daily_profit[i]
            df_second_way.box_required[y] = df_new.box_required[i]
            df_new2.daily_profit_loss[df_new.product_id[i]-1] = 0
        y += 1
    for j in range(len(df_couple.axes[0])):
        if df_new.product_id[i] == df_couple.first_prod[j]:
            if df_new2.product_id[df_couple.second_prod[j] - 1] == df_couple.second_prod[j] and df_new2.daily_profit_loss[df_couple.second_prod[j] - 1] != 0:
                if (no_of_total_boxes + df_new2.box_required[df_couple.second_prod[j] - 1]) <= 960:
                    df_second_way = df_second_way.append(df_temp2, ignore_index=True)
                    df_second_way.product_id[y] = df_new2.product_id[df_couple.second_prod[j] - 1]
                    df_second_way.daily_profit[y] = df_new2.daily_profit[df_couple.second_prod[j] - 1]
                    df_second_way.box_required[y] = df_new2.box_required[df_couple.second_prod[j] - 1]
                    df_new2.daily_profit_loss[df_couple.second_prod[j] - 1] = 0
                    no_of_total_boxes = no_of_total_boxes + df_new2.box_required[df_couple.second_prod[j] - 1]
                    y += 1
        if df_new.product_id[i] == df_couple.second_prod[j]:
            if df_new2.product_id[df_couple.first_prod[j] - 1] == df_couple.first_prod[j] and df_new2.daily_profit_loss[df_couple.first_prod[j] - 1] != 0:
                if (no_of_total_boxes + df_new2.box_required[df_couple.first_prod[j] - 1]) <= 960:
                    df_second_way = df_second_way.append(df_temp2, ignore_index=True)
                    df_second_way.product_id[y] = df_new2.product_id[df_couple.first_prod[j] - 1]
                    df_second_way.daily_profit[y] = df_new2.daily_profit[df_couple.first_prod[j] - 1]
                    df_second_way.box_required[y] = df_new2.box_required[df_couple.first_prod[j] - 1]
                    df_new2.daily_profit_loss[df_couple.first_prod[j] - 1] = 0
                    no_of_total_boxes = no_of_total_boxes + df_new2.box_required[df_couple.first_prod[j] - 1]
                    y += 1

df_second_way.to_csv('Warehouse_products_second_way.csv')

df_new2 = df_new2.sort_values(by=['product_id'])
df_new2 = df_new2.reset_index(drop=True)
df_new2.to_csv('final_list_for_loss_2nd_way.csv')

df_new = df_new.sort_values(by=['daily_profit_loss'], ascending=False)
df_new = df_new.reset_index(drop=True)
df_new.to_csv('final_list_3rd_way.csv')
values = df_new['daily_profit_loss']
weights = df_new['box_required']
c = 960
m = Model('Integer Problem')
n = len(df_new.axes[0])
x = m.addVars(n,vtype= GRB.BINARY, name = 'product')
m.setObjective(quicksum(values[i]* x[i] for i in range(n)), GRB.MAXIMIZE)
m.addConstr((quicksum(weights[i]* x[i] for i in range(n)) <= c), name="knapsack")

for i in range (len(df_couple.axes[0])):
	m.addConstr(x[df_couple.first_prod[i]] == x[df_couple.second_prod[i]])

m.setParam(GRB.Param.LogToConsole, 0)

m.optimize()
df_last = pd.DataFrame(columns=['product_id','value'])
for v in m.getVars():
	df_last.product_id[v] = v.varName
	df_last.value[v] = v.x

print(df_new.daily_profit_loss.sum() - m.objVal)
print(m.numVars)
df_last['value'] = m.getAttr(GRB.Attr.X, m.getVars())
df_last['product_id'] = m.getAttr(GRB.Attr.VarName, m.getVars())
for i in range(len(df_last.axes[0])):
	if df_last.value[i] == 0:
		df_last = df_last.drop(i)
df_last.to_csv('warehouse_products_3rd_way.csv')

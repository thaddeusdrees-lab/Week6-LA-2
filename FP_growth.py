
import pandas as pd
import mlxtend
import time
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

dataset = [['M','O','N','K','E','Y'],['D','O','N','K','E','Y'],['M','A','K','E'],['M','U','C','K','Y'],['C','O','K','I','E']]
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

start = time.perf_counter()
stuff = fpgrowth(df, min_support=0.6, use_colnames=True)
end = time.perf_counter()
print(end-start)
start = time.perf_counter()
things =apriori(df, min_support=0.6, use_colnames=True)
end = time.perf_counter()
print(end-start)
print(things,stuff)
table  = (association_rules(stuff, min_threshold=0.6))
print(table[(table['confidence'] != 1.00)])
print(table)

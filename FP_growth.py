import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

dataset = [['M','O','N','K','E','Y'],['D','O','N','K','E','Y'],['M','A','K','E'],['M','U','C','K','Y'],['C','O','K','I','E']]
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df
stuff = fpgrowth(df, min_support=0.6, use_colnames=True)
table  = (association_rules(stuff, min_threshold=0.6))
print(table[(table['confidence'] != 1.00)])
print(table)
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def question3() :
     dataset = pd.read_csv("DMHW1_data.csv")
     dataset = apriori(dataset, min_support=0.02, use_colnames=True)
     dataset = association_rules(dataset, metric='confidence', min_threshold=0.69)
     dataset = dataset[['antecedents', 'consequents', 'support', 'confidence']]
     pd.DataFrame(dataset).to_csv('output.csv')
     print("done, check output.csv plz")

def question4() :
     dataset = pd.read_csv("DMHW1_data.csv")
     dataset = apriori(dataset, min_support=0.0391, use_colnames=True)
     rule = association_rules(dataset, metric='confidence', min_threshold=0)
     rule = rule[['antecedents', 'consequents', 'support', 'confidence']]
     pd.DataFrame(rule).to_csv('output.csv')
     print("done, check output.csv plz")

def question5() :
     dataset = pd.read_csv("DMHW1_data.csv")
     dataset = apriori(dataset, min_support=0.01, use_colnames=True)
     rule = association_rules(dataset, metric='confidence', min_threshold=0)
     rule = rule[['antecedents', 'consequents', 'support', 'confidence']]
     rule['antecedents_len'] = rule['antecedents'].apply(lambda x: len(x))
     rule['consequents_len'] = rule['consequents'].apply(lambda x: len(x))
     rule = rule[((rule['antecedents_len'] + rule['consequents_len']) == 4) &
                 (rule['confidence'] > 0.9)]
     pd.DataFrame(rule).to_csv('output.csv')
     print("done, check output.csv plz")

def question6() :
     dataset = pd.read_csv("DMHW1_data.csv")
     dataset = apriori(dataset, min_support=0.01, use_colnames=True)
     rule = association_rules(dataset, metric='lift', min_threshold=50)
     rule = rule[['antecedents', 'consequents', 'support', 'lift']]
     pd.DataFrame(rule).to_csv('output.csv')
     print("done, check output.csv plz")



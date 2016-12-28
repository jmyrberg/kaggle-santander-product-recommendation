'''
Association rules algorithm for Santander Product Recommendation in Python.

Uses Yu Mochizuki's apryori.py

Created on 12.11.2016

@author: Jesse
'''
from data_utils import *
from evaluation import *
from apryori import apriori
from collections import defaultdict

def get_rules(X, min_support=1e-6):
    inds = (X > 0).as_matrix()
    cols = X.columns.values
    res = [cols[c].tolist() for c in inds]
    ap = list(apriori(res,min_support=min_support))
    l = []
    for r in ap:
        support = r.support
        for i in r.ordered_statistics:
            key = i.items_base
            value = i.items_add
            confidence = i.confidence
            lift = i.lift
            l.append([key,value,confidence,lift,support])
    return(l)

def dict2df(l):
    df = pd.DataFrame(l, columns=['source','target','confidence','lift','support'])
    df = df.groupby(['source','target']) \
        .aggregate({'confidence':sum,'lift':sum,'support':sum}) \
        .reset_index().sort_values(['source','confidence'], ascending=[True,False]) \
        .reset_index(drop=True)
    return(df)

def df2topdict(df):
    df = df.groupby('source')['target'].apply(lambda x: [list(e)[0] for e in x])
    d = df.to_dict()
    return(d)

def get_best(d,previous):
    return(d.get(frozenset(previous),d[frozenset([])]))

def fill_other(d_best,other=None):
    if other is None:
        other = ['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_ctop_fin_ult1', 'ind_ecue_fin_ult1', 'ind_cno_fin_ult1', 
               'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_ctpp_fin_ult1', 
               'ind_dela_fin_ult1', 'ind_valo_fin_ult1', 'ind_fond_fin_ult1', 'ind_ctma_fin_ult1', 'ind_plan_fin_ult1', 
               'ind_ctju_fin_ult1', 'ind_hip_fin_ult1', 'ind_viv_fin_ult1', 'ind_pres_fin_ult1', 'ind_deme_fin_ult1', 
               'ind_cder_fin_ult1', 'ind_deco_fin_ult1', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1']
    d_filled = {}
    for id,best in d_best.items():
        d_filled[id] = best + [e for e in other if e not in best]
    return(d_filled)

def diff_other(d_best,other):
    d_diffed = {}
    for id,best in d_best.items():
        d_diffed[id] = [e for e in best if e not in other.get(id,[])]
    return(d_diffed)
	
# Validate model
def validate():
    
    all_days = [1]
    
    mapk_avg = 0
    
    print('Loading sample...')
    x = load_large_sample()
    
    for n_days in all_days:
        
        x_train,x_last,d_last,x_test,d_test = split_train(x,n_days)
        
        X = x_train.set_index('ncodpers',drop=True,inplace=False)[prod_features].reset_index(drop=False,inplace=False).groupby('ncodpers').max()
        
        print('Finding rules...')
        d = get_rules(X)
        
        print('Converting to df...')
        df = dict2df(d)
        
        print('Getting top items for each source...')
        d = df2topdict(df)
        
        print('Filling dict with top items...')
        d = fill_other(d)
        
        print('Removing last items...')
        d = diff_other(d,d_last)
        
        print('Getting top for each user...')
        pred = []
        true = []
        for id,tr in d_test.items():
            previous = d_last.get(id,[])
            pred.append(get_best(d,previous))
            true.append(tr)
            
        res = mapk(true, pred)
        mapk_avg += res
        print('MAPK@7 (n_days=%d):\t%f' % (n_days,mapk(true,pred)))
        
    print('MAPK@7 average:\t%f' % (mapk_avg/len(all_days)))
    

# Run model for submission
def run_solution():
    
    filename = 'arules_submission.csv'
    
    print('Loading sample...')
    x_train,x_test = load_preprocessed()
    test_inds = x_test.ncodpers.values
    print(x_train.shape,x_test.shape)
    print(x_train.ncodpers.nunique(),x_test.ncodpers.nunique())
    
    print('Preprocessing...')
    X = x_train.set_index('ncodpers',drop=True,inplace=False)[prod_features].reset_index(drop=False,inplace=False).groupby('ncodpers').max()
    
    print('Forming x_last...')
    x_train.sort_values('fecha_dato',ascending=True,inplace=True)
    last_inds = ~x_train.duplicated('ncodpers',keep='last')
    x_last = x_train[last_inds]
    x_last.set_index('ncodpers',inplace=True,drop=True)

    print('Forming d_last...')
    last_ids = x_last.index.values
    inds = (x_last[prod_features] > 0.1).values
    cols = x_last[prod_features].columns.values
    d_last = [cols[take].tolist() for take in inds]
    d_last = dict((k,v) for k,v in zip(last_ids,d_last))

    print('Finding rules...')
    d = get_rules(X)
    
    print('Converting to df...')
    df = dict2df(d)
    
    print('Getting top items for each source...')
    d = df2topdict(df)
    
    print('Filling dict with top items...')
    d = fill_other(d)
    
    print('Removing last items...')
    d = diff_other(d,d_last)
        
    print('Getting top for each user...')
    pred = []
    for id in test_inds:
        previous = d_last.get(id,[])
        pred.append(get_best(d,previous)[:7])
    
    print('Writing to file...')
    with open(filename,'w',encoding='utf-8') as f:
        f.write('ncodpers' + "," + 'added_products' + "\n")
        for id,vec in zip(test_inds,pred):
            f.write(str(id) + "," + " ".join(vec) + "\n")
    print('Done!')
    
    print(pred[:100])
    
    

def main():
    validate()
    #run_solution()

if __name__ == '__main__':
    main()

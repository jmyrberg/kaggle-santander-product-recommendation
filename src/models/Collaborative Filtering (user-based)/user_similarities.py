'''
Recommends products based on top k similar users' experienced products.

Created on 28.10.2016

@author: Jesse
'''
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
from data_utils import *
from evaluation import *

pd.set_option('display.expand_frame_repr', None)
pd.set_option('display.max_columns', 40)

num_features = ['age','antiguedad','renta']
cat_features = ['segmento','tiprel_1mes']
prod_features = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',
                'ind_ctju_fin_ult1','ind_ctma_fin_ult1', 'ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
                'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
                'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
                'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
top = ['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_ctop_fin_ult1', 'ind_ecue_fin_ult1', 'ind_cno_fin_ult1', 
               'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_ctpp_fin_ult1', 
               'ind_dela_fin_ult1', 'ind_valo_fin_ult1', 'ind_fond_fin_ult1', 'ind_ctma_fin_ult1', 'ind_plan_fin_ult1', 
               'ind_ctju_fin_ult1', 'ind_hip_fin_ult1', 'ind_viv_fin_ult1', 'ind_pres_fin_ult1', 'ind_deme_fin_ult1', 
               'ind_cder_fin_ult1', 'ind_deco_fin_ult1', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1']
  
def transform_distance(D,metric):
    if metric in ['euclidean','minkowski','manhattan']:
        D = 1/(1+D)
    elif metric in ['cosine','jaccard','dice','matching']:
        D = 1-D
    else:
        D = 1/(1+D)
    np.fill_diagonal(D, 0)
    return(D)
  
def similarity(X1,X1_inds,X2,X2_inds,Y2,metric,k,combine_method,nlize=True,remove_zero_rows=True,**kwargs):
    D = pairwise_distances(X1, X2, metric=metric,**kwargs)

    D = transform_distance(D,metric)
    D[D>0.99] = 0 

    nn = np.argpartition(D, -k, axis=1)[:,-k:]
    D = D[np.arange(nn.shape[0])[:,None],nn]
    
    W = np.multiply(Y2[nn],D[:,:,None])
    
    if combine_method == 'sum':
        W = W.sum(1)
    elif combine_method == 'mean':
        W = W.mean(1)
    elif combine_method == 'max':
        W = W.max(1)
    elif combine_method == 'count':
        W[W>0] = 1
        W = W.sum(1)
        
    if nlize:
        W = normalize(W, norm='l1', axis=1)
        
    if remove_zero_rows:
        non_zero = W.sum(1) > 0
        W = W[non_zero]
        X1_inds = X1_inds[non_zero]
        
    d_probas = {}
    for id,w in zip(X1_inds,W):
        d_probas[id] = w
    return(d_probas)

def similarity_generator(X1,X1_inds,X2,X2_inds,Y2,metric='cosine',k=50,combine_method='sum',chunksize=1000,**kwargs):
    for i in range(0,X1.shape[0],chunksize):
                
                start = i
                end = np.min([i+chunksize,X1.shape[0]])
                
                print("Predicting proba for chunks %d - %d" % (start,end))
                sims = similarity(X1[start:end],X1_inds[start:end],X2,X2_inds,Y2,metric,k,combine_method,**kwargs)
                
                start += chunksize

                yield sims

def get_predictions(d,cols):
    prod = pd.DataFrame.from_dict(d, orient='index')
    probas = prod.as_matrix()
    best_inds = probas.argsort()[:,::-1]
    non_zero = (probas[np.arange(best_inds.shape[0])[:,None],best_inds] > 0)
    best = np.array(cols)[best_inds]
    best_list = [best[i][take].tolist() for i,take in enumerate(non_zero)]
    d_best = dict((k,v) for k,v in zip(prod.index.values,best_list))
    return(d_best)
            
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
          
def group_users(x_train,test_inds,prod_features):
    X2 = x_train.set_index('ncodpers',drop=True,inplace=False)[prod_features].reset_index(drop=False,inplace=False).groupby('ncodpers').max()
    X2 = X2[X2.sum(1) > 0]
    X2_inds = X2.index.values
    
    X1 = X2[X2.index.isin(test_inds)]
    X1_inds = X1.index.values
    
    X1 = X1.as_matrix()
    X2 = X2.as_matrix()
    return(X1,X1_inds,X2,X2_inds)
          
def validate():
    
    all_days = [4]
    
    mapk_avg = 0
    
    for n_days in all_days:
    
        print('\nLoading and transforming sample...')
        x = load_large_sample()
        x_train,x_last,d_last,x_test,d_test = split_train(x,n_days)
        
        # Top k similar users for each user based on experienced products
        print('Top k similar users for each user based on experienced products...')
        X1,X1_inds,X2,X2_inds = group_users(x_train, x_test.index.values, prod_features)
        prod_gen = similarity_generator(X1,X1_inds,X2,X2_inds,X2,
                                        metric='cosine',
                                        k=50,
                                        combine_method='sum',
                                        chunksize=5000)
        d_prod = {}
        for p in prod_gen:
            d_prod.update(p)
                
        # Combine similarities to predictions
        print('Converting similarities to predictions...')
        d_best = get_predictions(d_prod,prod_features)
        
        # Add top to the end
        d_best = fill_other(d_best)
        
        # Remove existing products
        d_pred = diff_other(d_best,d_last)
        
        pred = []
        true = [] 
        for id,tr in d_test.items():
            true.append(tr)
            pred.append(d_pred.get(id,[t for t in top if t not in d_last.get(id,[])]))
        res = mapk(true, pred)
        mapk_avg += res
        print('MAPK@7 (n_days=%d):\t%f' % (n_days,mapk(true,pred)))
        
    print('MAPK@7 average:\t%f' % (mapk_avg/len(all_days)))
    
def run_solution():
    
    print('\nLoading and transforming sample...')
    x_train,x_test = load_preprocessed()
    test_inds = x_test.ncodpers.values
    print(x_train.head())
    print(x_test.head())
    
    print('Forming x_last...')
    x_train.sort_values('fecha_dato',ascending=True,inplace=True)
    last_inds = ~x_train.duplicated('ncodpers',keep='last')
    x_last = x_train[last_inds]
    x_last.set_index('ncodpers',inplace=True,drop=True)
    print(x_last.head())
    
    top = x_last[prod_features].sum(0).sort_values().index.values[::-1].tolist()
    print(top)
    
    print('Forming d_last...')
    last_ids = x_last.index.values
    inds = (x_last[prod_features] > 0.1).values
    cols = x_last[prod_features].columns.values
    d_last = [cols[take].tolist() for take in inds]
    d_last = dict((k,v) for k,v in zip(last_ids,d_last))
    
    # Top k similar users for each user based on experienced products
    print('Top k similar users for each user based on experienced products...')
    X1,X1_inds,X2,X2_inds = group_users(x_train, test_inds, prod_features)
    prod_gen = similarity_generator(X1,X1_inds,X2,X2_inds,X2,
                                    metric='cosine',
                                    k=50,
                                    combine_method='sum',
                                    chunksize=1000)
    d_prod = {}
    for p in prod_gen:
        d_prod.update(p)
            
    # Combine similarities to predictions
    print('Converting similarities to predictions...')
    d_best = get_predictions(d_prod,prod_features)
    
    # Add top to the end
    print('Adding top to the end...')
    d_best = fill_other(d_best)
    
    # Remove existing products
    print('Removing existing products...')
    d_pred = diff_other(d_best,d_last)
    
    print('Forming predictions...')
    pred = []
    for id in test_inds:
        pred.append(d_pred.get(id,[t for t in top if t not in d_last.get(id,[])])[:7])
    
    print('Writing to file...')
    with open('user_similarities_20161112.csv','w',encoding='utf-8') as f:
        f.write('ncodpers' + "," + 'added_products' + "\n")
        for id,vec in zip(test_inds,pred):
            f.write(str(id) + "," + " ".join(vec) + "\n")
            
    print(pd.read_csv('user_similarities_20161112.csv',nrows=100).head(20))
    
def main():
    validate()
    #run_solution()
    
if __name__ == '__main__':
    main()
    
    
'''
Recommends products based on item similarities.

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
  
def transform_distance(D,metric):
    if metric in ['euclidean','minkowski','manhattan']:
        D = 1/(1+D)
    elif metric in ['cosine','jaccard','dice','matching']:
        D = 1-D
    else:
        D = 1/(1+D)
    np.fill_diagonal(D, 0)
    return(D)
  
def similarity(D,X2,X2_inds,combine_method,nlize=True,remove_zero_rows=True,**kwargs):    
    prods_to_take = X2 > 0
        
    D = [D[take].sum(0).tolist() for take in prods_to_take]

    d_probas = {}
    for id,d in zip(X2_inds,D):
        d_probas[id] = d
    return(d_probas)

def similarity_generator(X1,X2,X2_inds,metric='cosine',k=50,combine_method='sum',chunksize=1000,**kwargs):
    D = pairwise_distances(X1, X1, metric=metric,**kwargs)
    D = transform_distance(D,metric)
    for i in range(0,X2.shape[0],chunksize):
                
        start = i
        end = np.min([i+chunksize,X2.shape[0]])
        
        print("Predicting proba for chunks %d - %d" % (start,end))
        sims = similarity(D,X2[start:end],X2_inds[start:end],combine_method,**kwargs)
        
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
          
def get_processed_data(x_train,x_last,test_inds,prod_features):
    X1 = x_train.set_index('ncodpers',drop=True,inplace=False)[prod_features].reset_index(drop=False,inplace=False).groupby('ncodpers').max()
    X1 = X1[X1.sum(1) > 0]
    X1 = X1.as_matrix().T
    X2 = x_last.loc[x_last.index.isin(test_inds),prod_features]
    
    X2_inds = X2.index.values
    X2 = X2.as_matrix()
    #non_zero = X2.sum(1) > 0
    #X2 = X2[non_zero]
    #X2_inds = X2_inds[non_zero]
    return(X1,X2,X2_inds)
          
def validate():
    
    all_days = [3,2,1]
    
    mapk_avg = 0
    
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
    
    
    for n_days in all_days:
    
        print('\nLoading and transforming sample...')
        x = load_small_sample()
        x_train,x_last,d_last,x_test,d_test = split_train(x,n_days)
        
        # Top k similar users for each user based on experienced products
        print('Top k similar users for each user based on experienced products...')
        X1,X2,X2_inds = get_processed_data(x_train,x_last,x_test.index.values,prod_features)
        prod_gen = similarity_generator(X1,X2,X2_inds,
                                        metric='cosine',
                                        chunksize=5000)
        d_prod = {}
        for p in prod_gen:
            d_prod.update(p)
        df = pd.DataFrame.from_dict(d_prod, orient='index')
        print(df)
                
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
    
def main():
    validate()
    
if __name__ == '__main__':
    main()
    
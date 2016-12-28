'''
XGBoost for product recommendation.

Created on 4.12.2016

@author: Jesse
'''
from data_utils import *
from evaluation import *
import operator, gc
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from keras.preprocessing.sequence import pad_sequences

def create_train_data(x,date):
    print('Creating training data for date %s...' % date)
    pydate = get_previous_year(date)
    dates = get_dates_between(get_year_start(pydate),pydate)
    n_dates = len(dates)
    print('dates',dates)
    
    d_test = load_target(pydate)
    
    prods = prod_features()
    n_prods = len(prods)
    use_ids = []
    for k,v in d_test.items():
        if len(v) > 0:
            use_ids.append(k)
    print(x.fecha_dato.value_counts())   
    x = x[x.fecha_dato.isin(dates)]
    x = x[x.ncodpers.isin(use_ids)]
    cols = [e for e in x.columns if e not in ['fecha_dato','fecha_alta']]
    n_cols = len(cols)
    x['temp'] = x[cols].values.tolist()#x.groupby(['ncodpers','fecha_dato']).apply(lambda x: list(x.values)).reset_index('fecha_dato') )
    x = x.groupby('ncodpers')['temp'].apply(lambda x: list(x))
    ncodpers = x.index.values
    x = pad_sequences(x.values, maxlen=n_dates, dtype=np.float32)
    
    y = []
    for id in ncodpers:
        inds = [prods.index(p) for p in d_test[id]]
        vec = np.zeros(n_prods,dtype=np.int16)
        vec[inds] = 1
        y.append(vec)
    y = np.reshape(y, (len(y),n_prods))
    
    assert x.shape[0] == y.shape[0]
    
    folder = './xgb/data/' + date + '/'
    path = folder+'xy'
    create_folder(folder)
    np.savez(folder+'xy',x,y)
    print(x.shape,y.shape)
    print('Train data saved in %s!' % (path+'.npz'))
    
def load_train_data(date):
    f = np.load('./xgb/data/' + date + '/' + 'xy.npz')
    x = f['arr_0']
    y = f['arr_1']
    return(x,y)

def create_test_data(x,date):
    print('Creating test data for date %s...' % date)
    dates = get_dates_between(get_year_start(date),date)
    n_dates = len(dates)
    print('dates',dates)
    
    d_test = load_target(date)
    
    if d_test is None:
        test_ids = pd.read_csv('../data/raw/sample_submission.csv',usecols=['ncodpers']).ncodpers.values
        print(test_ids)
    else:
        test_ids = list(d_test.keys())
    print(x.fecha_dato.value_counts())   
    x = x[x.fecha_dato.isin(dates)]
    x = x[x.ncodpers.isin(test_ids)]
    print(x.ncodpers.nunique())
    cols = [e for e in x.columns if e not in ['fecha_dato','fecha_alta']]
    n_cols = len(cols) # ncodpers and date
    x['temp'] = x[cols].values.tolist()#x.groupby(['ncodpers','fecha_dato']).apply(lambda x: list(x.values)).reset_index('fecha_dato') )
    x = x.groupby('ncodpers')['temp'].apply(lambda x: list(x))
    print(x.shape)
    print('d',x.index.nunique())
    tids = x.index.values
    print(tids)
    x = pad_sequences(x.values, maxlen=n_dates, dtype=np.float32)
    print(x.shape)
    
    folder = './xgb/data/' + date + '/'
    path = folder+'xt'
    create_folder(folder)
    np.savez(folder+'xt',x)
    with open(folder+'xt_id.list','wb') as f:
        pickle.dump(tids,f)
    print('Test data saved in %s and %s!' % (path+'.npz',folder+'xt_id.list'))
    print(x.shape)

def load_test_data(date):
    folder = './xgb/data/' + date + '/'
    with open(folder+'xt_id.list','rb') as f:
        tids = pickle.load(f)
    xt = np.load(folder + 'xt.npz')['arr_0']
    return(xt,tids)

def create_datasets():
    x = load_preprocessed()
    dates = ['2016-06-28']#,'2016-06-28']#['2016-04-28','2016-05-28','2016-06-28']
    for date in dates:
        print(date)
        create_train_data(x, date)
        gc.collect()
        create_test_data(x, date)
        gc.collect()
        
def load_dataset(date):
    x,y = load_train_data(date)
    xt,tids = load_test_data(date)
    d_last = load_last(date)
    d_test = load_target(date)
    return(x,y,xt,tids,d_last,d_test)

def get_model(x,y):
    xgb_params = {}
    xgb_params["nthread"] = 4
    xgb_params["objective"] = "multi:softprob"
    xgb_params["eval_metric"] = ["mlogloss"]
    xgb_params["learning_rate"] = 0.1
    xgb_params["subsample"] = 0.8
    xgb_params["colsample_bytree"] = 0.8
    xgb_params["silent"] = 1
    xgb_params["num_class"] = 22
    xgb_params["max_depth"] = 6
    xgb_params["min_child_weight"] = 12
    xgb_params["seed"] = 3030
    return(xgb_params)

def process_dataset(x,xt,y):
    
    print(x.shape,xt.shape,y.shape)
    print(x[1000])
    
    cols_to_del = []
    for i in range(x.shape[2]):
        min_val = np.min([x[:,:,i].min(),xt[:,:,i].min()])
        max_val = np.max([x[:,:,i].max(),xt[:,:,i].max()])
        rng = max_val - min_val
        if rng > 0.001:
            continue
        else:
            cols_to_del.append(i)

    x = np.delete(x,cols_to_del,axis=2)
    xt = np.delete(xt,cols_to_del,axis=2)

    x_vars = []
    targets = []
    for i,row in enumerate(y):
        row = row[2:]
        for j,prod in enumerate(row):
            if prod > 0:
                x_vars.append(x[i])
                targets.append(j)
                
    # Reshape
    x = np.reshape(x_vars,(len(x_vars),x.shape[1]*x.shape[2]))
    xt = np.reshape(xt,(xt.shape[0],xt.shape[1]*xt.shape[2]))
    y = np.array(targets)
    
    #x = x[:,-102:]
    #xt = xt[:,-102:]
    
    #x = proc(x)
    #xt = proc(xt)
    print(x.shape,xt.shape)    
    return(x,xt,y)

def get_probas(date,cv):
    print('Fitting and getting predictions...')
    x,y,xt,tids,d_last,d_test = load_dataset(date)
    
    x,xt,y = process_dataset(x,xt,y) 
    print(x.shape,y.shape,xt.shape)

    print('Fitting...')
    xgb_params = get_model(x,y)
    num_boost_round = 60
    if cv:
        test_size = 0.05
        _x, _xt, _y, _yt = train_test_split(x,y,test_size=test_size)
        _dx = xgb.DMatrix(_x,label=_y,missing=-999.)
        _dxt = xgb.DMatrix(_xt,label=_yt,missing=-999.)
        res = xgb.train(xgb_params, _dx, num_boost_round=300, early_stopping_rounds=10, evals=[(_dx,'train'),(_dxt,'eval')], maximize=False, verbose_eval=True)
        num_boost_round = int(round(res.best_ntree_limit / (1-test_size)))
        print('Best number of rounds',num_boost_round)
        
    dx = xgb.DMatrix(x,label=y,missing=-999.)
    dxt = xgb.DMatrix(xt,missing=-999.)
    clf = xgb.train(xgb_params, dx, num_boost_round=num_boost_round, evals=[(dx,'train')], maximize=True, verbose_eval=True)
    #xgb.plot_importance(clf)
    scores = clf.get_fscore()
    sorted_x = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    sorted_list = [int(k[1:]) for k,v in sorted_x]
    print(sorted_x)
    print(sorted_list)
    plt.show()
    
    print('Predicting...')
    probas = clf.predict(dxt)
    print(probas)
    return(probas,tids,d_last,d_test)

def get_preds(date,cv):
    top = ['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_cno_fin_ult1', 'ind_ecue_fin_ult1', 
           'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_ctop_fin_ult1', 'ind_reca_fin_ult1', 
           'ind_tjcr_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_dela_fin_ult1', 'ind_valo_fin_ult1', 
           'ind_fond_fin_ult1', 'ind_ctma_fin_ult1', 'ind_plan_fin_ult1', 'ind_deco_fin_ult1', 
           'ind_hip_fin_ult1', 'ind_viv_fin_ult1', 'ind_deme_fin_ult1', 'ind_ctju_fin_ult1', 'ind_pres_fin_ult1', 
           'ind_cder_fin_ult1', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1']
    probas,tids,d_last,d_test = get_probas(date,cv)
    prods = np.array(prod_features())[2:]
    pred = prods[probas.argsort()[:,::-1]]
    d_pred = dict((id,val) for id,val in zip(tids,pred))
    return(d_pred,d_last,d_test,top)

def validate(cv=False): # 0.027832 / 0.0299399, 0.028032 / 0.0298XXX, 0.028132 / 0.029299, 028109 / 0.029899
    print('Validating...')
    
    date = '2016-05-28'
    
    d_pred,d_last,d_test,top = get_preds(date,cv)
    
    preds = []
    trues = []
    for id,tr in d_test.items():
        tr = d_test[id]
        trues.append(tr)
        pred = [p for p in d_pred.get(id,top) if p not in d_last.get(id,[])][:7]
        preds.append(pred)
        #if len(tr) > 0 and len(d_last[id]) > 0:
        #    print('\nTrue',tr)
        #    print('Existing',d_last[id])
        #    print('Pred',pred)
    print('\nDate %s: %6f' % (date,round(mapk(trues,preds),6)))

def run_solution(cv=False):
    d_pred,d_last,d_test,top = get_preds('2016-06-28',cv)
    test_ids = pd.read_csv('../data/raw/sample_submission.csv',usecols=['ncodpers']).ncodpers.values
    d_pred2 = {}
    for id in test_ids:
        pred = [p for p in d_pred.get(id,top) if p not in d_last.get(id,[])][:7]
        d_pred2[id] = pred
    assert len(d_pred2) == len(test_ids)
    write_submission(d_pred2,'xgb_final_try1_submission.csv')

def bag_model():
    
    date = '2016-06-28'
    n_bags = 10
    test_size = 0.05
    folder = './xgb/data/bagged/'
    
    create_folder(folder)
    prods = np.array(prod_features())[2:]
    n_prods = prods.shape[0]
    
    x,y,xt,tids,d_last,d_test = load_dataset(date)
    x,xt,y = process_dataset(x,xt,y)
    dxt = xgb.DMatrix(xt)
    xgb_params = get_model(x,y)
        
    n_test = len(tids)
    kf = KFold(n_bags)
    
    yt = np.zeros((n_test,n_prods))
    yoof = np.zeros((x.shape[0],n_prods))
    for i,(tri,tei) in enumerate(kf.split(x)):
        print('Bag',i+1)
        
        xf,yf,xtf,ytf = x[tri],y[tri],x[tei],y[tei]
        
        # Find optimal n_rounds
        _x, _xt, _y, _yt = train_test_split(xf,yf,test_size=test_size)
        _dx = xgb.DMatrix(_x,label=_y)
        _dxt = xgb.DMatrix(_xt,label=_yt)
        res = xgb.train(xgb_params, _dx, num_boost_round=200, early_stopping_rounds=10, evals=[(_dx,'train'),(_dxt,'eval')], maximize=False, verbose_eval=True)
        num_boost_round = int(round(res.best_ntree_limit / (1-test_size)))
        
        # Fit
        dxf = xgb.DMatrix(xf,label=yf)
        clf = xgb.train(xgb_params, dxf, num_boost_round=num_boost_round, evals=[(dxf,'train')], maximize=True, verbose_eval=True)
        
        # Predict out of fold and also test set
        dxtf = xgb.DMatrix(xtf)
        yoof[tei] = clf.predict(dxtf)
        yt += clf.predict(dxt)
        
    yoof /= n_bags
    print(yoof)
    print(yt)
    print(yoof.shape,yt.shape)
    folder += date+'/'
    create_folder(folder)
    np.savez(folder+'yoof',yoof)
    np.savez(folder+'yt',yt)
    
    pred = prods[yt.argsort()[:,::-1]]
    d_pred = dict((id,val) for id,val in zip(tids,pred))
    if date=='2016-06-28':
        d_pred2 = {}
        for k,v in d_pred.items():
            d_pred2[k] = [e for e in v if e not in d_last.get(k,[])][:7]
        write_submission(d_pred2,'bagged_' + str(n_bags) + '_xgb2.csv')
    else:
        preds = []
        trues = []
        for id,tr in d_test.items():
            top = ['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_cno_fin_ult1', 'ind_ecue_fin_ult1', 
                   'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_ctop_fin_ult1', 'ind_reca_fin_ult1', 
                   'ind_tjcr_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_dela_fin_ult1', 'ind_valo_fin_ult1', 
                   'ind_fond_fin_ult1', 'ind_ctma_fin_ult1', 'ind_plan_fin_ult1', 'ind_deco_fin_ult1', 
                   'ind_hip_fin_ult1', 'ind_viv_fin_ult1', 'ind_deme_fin_ult1', 'ind_ctju_fin_ult1', 'ind_pres_fin_ult1', 
                   'ind_cder_fin_ult1', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1']
            tr = d_test[id]
            trues.append(tr)
            pred = [p for p in d_pred.get(id,top) if p not in d_last.get(id,[])][:7]
            preds.append(pred)
        print('\nDate %s: %6f' % (date,round(mapk(trues,preds),6)))
    

def main():
    create_datasets()
    #validate(False)
    #run_solution() # Single xgb submission
    bag_model() # Bagged xgb submission
    
if __name__ == '__main__':
    main()
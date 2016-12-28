'''
Long-Short Term Memory Neural Network for product recommendation.

Created on 14.12.2016

@author: Jesse
'''
from data_utils import *
from evaluation import * 
import pandas as pd
import csv
import pickle
from sklearn.model_selection import KFold
from collections import defaultdict
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.preprocessing.sequence import pad_sequences

def create_seqs():
    x,_ = load_raw()
    x = x[['ncodpers','fecha_dato'] + prod_features()]
    x[prod_features()] = x[prod_features()].astype(np.float32)
    
    seqs = defaultdict(list)
    prods = np.array(prod_features())
    
    counter = 0
    for i,(nc,g) in enumerate(x.groupby('ncodpers')):
        
        counter += 1
        
        change = (g[prods] - g[prods].shift(1)).values.astype(np.int8)
        change[change<0] = 0
        inds = np.arange(change.shape[0])[np.any(change>0,axis=1)]
        
        if len(inds) > 0:
        
            fecha_dato = g.fecha_dato.values
        
            for ind in inds:
                x = [(nc,v) for v in fecha_dato[:(ind+1)]]
                y = change[ind]
                seqs[fecha_dato[ind]].append((x,y))
                
        if counter % 10000 == 0:
            counter = 0
            print(i)
        
    folder = './data/lstm2/seqs/'
    create_folder(folder)
    with open(folder+'seqs.dict','wb') as f:
        pickle.dump(seqs,f)
    print('Saved sequences to %s!' % folder)
    
def create_datasets():
    
    df = load_preprocessed()
    print(df.fecha_dato.value_counts())
    df = df.set_index(['ncodpers','fecha_dato'],drop=True)
    df = df.select_dtypes(exclude=['datetime','object'])
    print(df.columns)
  
    # Train data
    train_dates = [d for d in get_dates_between('2015-02-28', '2016-05-28')] # Choose dates that are used as train data
    
    folder = './data/lstm2/seqs/'
    with open(folder+'seqs.dict','rb') as f:
        din = pickle.load(f)
    d = {}
    maxlen = 0
    for date in train_dates:
        x,y,nc = [],[],[]
        n = len(din[date])
        d[date] = {}
        for i,(X,Y) in enumerate(din[date]):
            vec = df.loc[X].values
            maxlen = max(maxlen,vec.shape[0])
            nc.append(X)
            x.append(vec)
            y.append(Y)
            print("Train date %s: %d / %d" % (date,i+1,n))
        
        d[date]["nc"] = np.array(nc)
        d[date]["x"] = pad_sequences(x,maxlen=maxlen)
        d[date]["y"] = np.array(y)
    
    file = folder+'train.dict'
    with open(file,'wb') as f:
        pickle.dump(d,f)
    print('Saved to %s!' % file)
    
    create_test_data(df) 
    
def create_test_data(x):
    gc.collect()
    print('Creating test data...')
    dates = get_dates_between('2015-01-28','2016-06-28') # Choose dates that are used as test data
    n_dates = len(dates)
    folder = './data/lstm2/seqs/'
    test_ids = pd.read_csv('../data/raw/sample_submission.csv',usecols=['ncodpers']).ncodpers.values
    x = x.reset_index().drop('fecha_dato',axis=1)
    print(x.columns)
    gc.collect()

    x = x[x.ncodpers.isin(test_ids)]
    print(x.ncodpers.nunique())
    cols = x.columns
    x['temp'] = x[cols].values.tolist()
    n_cols = len(cols)
    x = x.groupby('ncodpers')['temp'].apply(lambda x: list(x))
    print(x.shape)
    print('d',x.index.nunique())
    tids = x.index.values
    print(tids)
    x_temp = []
    print(len(x.values))
    for r in x.values:
        seq = np.array(r)
        x_temp.append(np.vstack((np.zeros((n_dates-seq.shape[0],seq.shape[1])),seq)))
    gc.collect()
    del x
    print(len(x_temp))
    x = np.reshape(x_temp, (len(x_temp),n_dates,n_cols))
    print(x.shape)
    
    path = folder+'xt'
    create_folder(folder)
    np.savez(folder+'xt',x)
    with open(folder+'xt_id.list','wb') as f:
        pickle.dump(tids,f)
    print('Test data saved in %s and %s!' % (path+'.npz',folder+'xt_id.list'))
    print(x.shape)
    
    
def get_score(d_pred,d_test,d_last):
    top = ['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_cno_fin_ult1', 'ind_ecue_fin_ult1', 
           'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_ctop_fin_ult1', 'ind_reca_fin_ult1', 
           'ind_tjcr_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_dela_fin_ult1', 'ind_valo_fin_ult1', 
           'ind_fond_fin_ult1', 'ind_ctma_fin_ult1', 'ind_plan_fin_ult1', 'ind_deco_fin_ult1', 
           'ind_hip_fin_ult1', 'ind_viv_fin_ult1', 'ind_deme_fin_ult1', 'ind_ctju_fin_ult1', 'ind_pres_fin_ult1', 
           'ind_cder_fin_ult1', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1']
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
    score = mapk(trues,preds)
    print('MAPK: %6f' % score)
    return(score)

def get_preds(test_date):
    
    nc,x,y,nct,xt,yt,d_last,d_test = load_datasets(test_date)
    
    x,xt = process_data(x,xt)
    
    clf = get_model(x,y)
    
    clf.fit(x,y,nb_epoch=5,batch_size=16)
    
    probas = clf.predict(xt)
    
    preds = np.array(prod_features())[probas.argsort()[:,::-1]]
    d_pred = dict((k,v) for k,v in zip(nct,preds))
    return(d_pred,d_test,d_last)
    
def load_datasets(test_date):
    print('Loading datasets...')
    folder = './data/lstm2/seqs/'
    with open(folder+'train.dict','rb') as f:
        d = pickle.load(f)
    
    # Predicting June 2015 with different sets of train data, validation results:
    # 04: MAPK: 0.041648
    # 05: MAPK: 0.040860
    # 12: MAPK: 0.044356
    # 01: MAPK: 0.041397
    # 02: MAPK: 0.041776
    # 03: MAPK: 0.040924
    # 04: MAPK: 0.042373
    # 05: MAPK: 0.040734
    # 12 + 04(6): MAPK: 0.045747
    # 12 + 04(6) + 05(6): MAPK: 0.047088
    # 12 + 04(6) + 05(6) + 01(6): MAPK: 0.047375
    # 12 + 04(6) + 05(6) + 01(6) + 02(6): MAPK: 0.046424
    
    # Choose train data from all available dates
    train_dates = ['2015-12-28','2015-06-28']#['2015-12-28','2016-04-28','2016-05-28','2016-01-28', '2016-02-28']#[da for da in get_dates_between('2015-02-28', '2016-05-28') if d != test_date]
    nc,x,y = [],[],[]
    for date in train_dates:
        print(date)
        nc.extend(list(d[date]["nc"]))
        x.extend(list(d[date]["x"]))
        y.extend(list(d[date]["y"]))
        print(len(x))
    if test_date != '2016-06-28':
        nct,xt,yt = list(d[test_date]["nc"]),list(d[test_date]["x"]),list(d[test_date]["y"])
        nct = [e[0][0] for e in nct]
    else:
        with open(folder+'xt_id.list','rb') as f:
            nct = pickle.load(f)
        xt = np.load(folder+'xt.npz')['arr_0'][:,:,1:]
        yt = None
    
    xt = pad_sequences(xt,maxlen=xt[0].shape[0])
    x = pad_sequences(x,maxlen=xt[0].shape[0])
    yt = np.array(yt)
    y = np.array(y)
    
    d_last = load_last(test_date)
    d_test = load_target(test_date)
    if d_test is None:
        d_test = pd.read_csv('../data/raw/sample_submission.csv',usecols=['ncodpers']).ncodpers.values
    print(x.shape,y.shape,xt.shape,yt.shape,len(d_last),len(d_test))
    return(nc,x,y,nct,xt,yt,d_last,d_test)

    
def get_model(x,y):
    print('Building LSTM model...')
    model = Sequential()
    model.add(LSTM(32, input_shape=(x.shape[1],x.shape[2]), return_sequences=True, stateful=False))
    model.add(LSTM(16, input_shape=(x.shape[1],x.shape[2]), return_sequences=False, stateful=False))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam') #
    return(model)

def process_data(x,xt):
    x[x==-1] = 0
    xt[xt==-1] = 0
    cols_to_del = []
    for i in range(x.shape[2]):
        min_val = np.min([x[:,:,i].min(),xt[:,:,i].min()])
        max_val = np.max([x[:,:,i].max(),xt[:,:,i].max()])
        rng = max_val - min_val
        if rng > 0.001:
            x[:,:,i] = (x[:,:,i] - min_val) / rng 
            xt[:,:,i] = (xt[:,:,i] - min_val) / rng
        else:
            cols_to_del.append(i)

    x = np.delete(x,cols_to_del,axis=2)
    xt = np.delete(xt,cols_to_del,axis=2)
    print(x[0,0,0])
    print(xt[0,0,0])
    
    x = x[:,-6:,:]
    xt = xt[:,-6:,:]
    print(x.shape,xt.shape)
    return(x,xt)

def validate():
    
    date = '2016-06-28'
    
    d_pred,d_test,d_last = get_preds(date)
    
    if date != '2016-06-28':
        get_score(d_pred,d_test,d_last)
    else:
        top = ['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_cno_fin_ult1', 'ind_ecue_fin_ult1', 
           'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_ctop_fin_ult1', 'ind_reca_fin_ult1', 
           'ind_tjcr_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_dela_fin_ult1', 'ind_valo_fin_ult1', 
           'ind_fond_fin_ult1', 'ind_ctma_fin_ult1', 'ind_plan_fin_ult1', 'ind_deco_fin_ult1', 
           'ind_hip_fin_ult1', 'ind_viv_fin_ult1', 'ind_deme_fin_ult1', 'ind_ctju_fin_ult1', 'ind_pres_fin_ult1', 
           'ind_cder_fin_ult1', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1']
        preds = []
        for id in d_test:
            pred = [p for p in d_pred.get(id,top) if p not in d_last.get(id,[])][:7]
            preds.append(pred)
        d_pred = dict((k,v) for k,v in zip(d_test,preds))
        write_submission(d_pred,'lstm_submission.csv')
        

        
def main():
    create_datasets()
    validate()
    
if __name__ == '__main__':
    main()
    
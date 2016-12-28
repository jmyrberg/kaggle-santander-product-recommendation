# -*-  coding: utf8 -*-
'''
Helpers and preprocessing methods.

Created on 26.11.2016

@author: Jesse Myrberg
'''
import pandas as pd
import numpy as np
import gc
import pickle
import os
import csv
from os.path import isfile, exists
from os import listdir, makedirs
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelBinarizer,OneHotEncoder
from collections import defaultdict
from scipy.stats import rankdata

pd.set_option('display.expand_frame_repr', None)

#### HELPERS ####
def write_submission(d_pred,filename):
    print('Writing to file...')
    with open(filename,'w',encoding='utf-8') as f:
        f.write('ncodpers' + "," + 'added_products' + "\n")
        for id,pred in d_pred.items():
            f.write(str(id) + "," + " ".join(pred) + "\n")
    print('File %s written!' % filename)

def bp():
    return(os.path.join(os.path.dirname(__file__)))

def get_previous_month(date):
    previous_date = str(((pd.to_datetime(date) - pd.DateOffset(months=1)).to_pydatetime().date()))
    return(previous_date)

def get_previous_year(date):
    previous_date = str(((pd.to_datetime(date) - pd.DateOffset(months=12)).to_pydatetime().date()))
    return(previous_date)

def get_dates_between(min_date,max_date):
    dates = ['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28',
             '2015-06-28','2015-07-28','2015-08-28','2015-09-28','2015-10-28',
             '2015-11-28','2015-12-28','2016-01-28','2016-02-28','2016-03-28',
             '2016-04-28','2016-05-28','2016-06-28']
    min_ind = dates.index(min_date)
    max_ind = dates.index(max_date)+1
    return(dates[min_ind:max_ind])

def get_year_start(date):
    if '2015' in date:
        return('2015-01-28')
    else:
        return('2016-01-28')

def create_folder(path):
    if not exists(path):
        makedirs(path)
        print('Created folder %s' % path)
        
		
#### LOADING ####
def load_raw():
    x_train = pd.read_csv(bp()+'/data/raw/train_ver2.csv', encoding='utf8')
    x_test = pd.read_csv(bp()+'/data/raw/test_ver2.csv', encoding='utf8')
    return(x_train,x_test)

def load_preprocessed():
    x = pd.read_csv(bp()+'/data/preprocessed/x.csv', encoding='utf8', dtype=dtypes())
    x['fecha_dato'] = pd.to_datetime(x.fecha_dato,format='%Y-%m-%d')
    return(x)

def load_target(date):
    if date != '2016-06-28':
        with open(bp() + '/data/targets/' + date + '/d_test.dict','rb') as f:
            d = pickle.load(f)
        return(d)
    else:
        print('No target for %s!' % date)
        return(None)

def load_last(date):
    with open(bp() + '/data/last/' + date + '/d_last.dict','rb') as f:
        d = pickle.load(f)
    return(d)
	
def load_small_sample():
    print('Loading small sample...')
    x = pd.read_csv('./data/samples/x_small_sample.csv', dtype=dtypes)
    return(x)

def load_large_sample():
    print('Loading large sample...')
    x = pd.read_csv('./data/samples/x_large_sample.csv', dtype=dtypes)
    return(x)
	
# Load multiple datasets (for validation)
def split_train(x,n_days=1):
    print('Splitting train...')
    x.sort_values('fecha_dato',ascending=True, inplace=True)
    x['fecha_dato'] = pd.to_datetime(x.fecha_dato)
    
    cols = x.iloc[:,-24:].columns
    test_days = np.sort(x.fecha_dato.unique())[-n_days:]
    test_day = [test_days[0]]
    train_inds = ~x.fecha_dato.isin(test_days)
    test_inds = x.fecha_dato.isin(test_day)
    
    x_train = x[train_inds]
    x_test = x[test_inds]
    
    test_ids = x_test.ncodpers.unique()
    
    last_inds = ~x_train.duplicated('ncodpers',keep='last')
    x_last = x_train[last_inds]
    x_last.set_index('ncodpers',inplace=True,drop=True)
    
    last_ids = x_last.index.values
    
    inds = (x_last.iloc[:,-24:] > 0.1).values.tolist()
    d_last = [cols[take].values.tolist() for take in inds]
    d_last = dict((k,v) for k,v in zip(last_ids,d_last))
    
    inds = (x_test.iloc[:,-24:] > 0.1).values.tolist()
    d_test = [set(cols[take].values.tolist()) for take in inds]
    d_test = dict((k,v) for k,v in zip(test_ids,d_test))
    d_test = dict((id,list(e.difference(d_last.get(id,[])))) for id,e in d_test.items())
    
    x_test = x_test.iloc[:,:-24]
    #x_test.set_index('ncodpers',inplace=True,drop=True)
    
    return(x_train,x_last,d_last,x_test,d_test)

	
#### DATA CREATION ####

# Create targets for a specific date (customers who have bought something on a specific date)
def create_target(x,date):
    print('Creating target for %s...' % date)
    previous_date = get_previous_month(date)
    x = x[x.fecha_dato.isin([previous_date,date])]
    x = x.sort_values(['ncodpers','fecha_dato'])
    print('number of unique values',x[x.fecha_dato==date].ncodpers.nunique())
    
    prods = np.array(prod_features())
    
    ncodpers = x.loc[x.fecha_dato==date,"ncodpers"].values
    prev_ncodpers = x.loc[x.fecha_dato==previous_date,"ncodpers"].values
    both_ncodpers = set(ncodpers).intersection(prev_ncodpers)
    one_ncodpers = set(ncodpers).difference(prev_ncodpers)
    print('sum',len(both_ncodpers)+len(one_ncodpers))
    
    #print('in both',both_ncodpers)
    xg = x[x.ncodpers.isin(both_ncodpers)]
    xg_previous = xg[xg.fecha_dato==previous_date].set_index('ncodpers').sort_index()
    xg_previous = xg_previous[prods]
    xg_current = xg[xg.fecha_dato==date].set_index('ncodpers').sort_index()
    xg_current = xg_current[prods]
    xg = xg_current - xg_previous
    xg[xg < 0] = 0

    #print('in later only',one_ncodpers)
    nx = x.loc[x.ncodpers.isin(one_ncodpers)].set_index('ncodpers')
    nx = nx[prods].astype('int16')
    
    x = pd.concat([nx,xg],axis=0)
    
    indicators = (x > 0).as_matrix()
    prods = x[prod_features()].columns
    d = dict((id,prods[best]) for id,best in zip(x.index,indicators))

    folder = bp() + '/data/targets/' + date + '/'
    create_folder(folder)
    with open(folder + 'd_test.dict', 'wb') as f:
        pickle.dump(d,f)
    print('Targets saved to %s!' % (folder + 'd_test.dict'))
    print('Number of unique values at end',len(d.keys()))
    
def create_targets():
    x = load_preprocessed()
    dates = [date for date in x.fecha_dato.unique() if date not in ['2015-01-28','2016-06-28']]
    for date in dates:
        create_target(x,date)
        
# Create information about the items that the customer already had on a specific date
def create_last(x,date):
    print('Creating last for %s...' % date)
    previous_date = get_previous_month(date)
    x = x[x.fecha_dato<=previous_date]
    
    x = x.sort_values('fecha_dato')
    last_inds = ~x.duplicated('ncodpers',keep='last')
    x = x[last_inds]
   
    ncodpers = x.ncodpers.values
    last = x[prod_features()].values > 0
    prods = np.array(prod_features())
    
    d = defaultdict(list)
    for k,v in zip(ncodpers,last):
        d[k] = prods[v].tolist()
    folder = bp() + '/data/last/' + date + '/'
    create_folder(folder)
    path = folder + 'd_last.dict'
    with open(path,'wb') as f:
        pickle.dump(d,f)
    print('Last saved in %s!' % path)
    
def create_lasts():
    x = load_preprocessed()
    dates = x.fecha_dato.unique()
    for date in dates:
        create_last(x,date)


# Return column names / data types for columns
def prod_features():
    prod_features = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1',
                    'ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
                    'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
                    'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
                    'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
    return(prod_features)

def dtypes():
    with open(bp()+'/data/preprocessed/dtypes.dict','rb') as f:
        dtypes = pickle.load(f)
    dtypes['fecha_dato'] = 'object'
    dtypes['fecha_alta'] = 'object'
    dtypes['ult_fec_cli_1t'] = 'object'
    return(dtypes)

# Generic method for creating preprocessed dataset from raw train and test data
# Change "preprocess_final" to some other function, if you want the raw data to be processed differently
def create_preprocessed():
    print('Creating processed...')
    x_train,x_test = load_raw()
    x = pd.concat([x_train,x_test])
    del x_train,x_test
    x.sort_values(['ncodpers','fecha_dato'],inplace=True)
    x.reset_index(inplace=True,drop=True)
    x = preprocess_final(x)
    x.sort_values(['ncodpers','fecha_dato'],inplace=True)
    x.reset_index(inplace=True,drop=True)
    folder = bp()+'/data/preprocessed/'
    create_folder(folder)
    d_dtypes = x.dtypes.to_dict()
    with open(folder+'dtypes.dict','wb') as f:
        pickle.dump(d_dtypes,f)
    x.to_csv(folder+'x.csv', encoding='utf8', index=False)
    print('Saved processed!')
    
# Create sample for preprocessed, used for some models
def create_sample(n_users,filename):
    x_train,_ = load_preprocessed()
    users = x_train['ncodpers'].unique()
    users = np.random.choice(users, size=n_users, replace=False)
    x_sample = x_train[x_train['ncodpers'].isin(users)]
	create_folder('./data/samples/')
    x_sample.to_csv('./data/samples/' + filename, index=False, encoding='utf8')
    
def create_small_sample(n_users=20000,filename='x_small_sample.csv'):
    create_sample(n_users,filename)
    
def create_large_sample(n_users=100000,filename='x_large_sample.csv'):
    create_sample(n_users,filename)
	
# Preprocessing function considered to be one of the best in the end
def preprocess_final(x):
    
    startshape = x.shape[0]
    print('Start shape',startshape)
    
    # Take original NaN count for every row
    print('n_nan')
    x['n_nan'] = x.isnull().sum(1)
    x['n_nan'] = x.n_nan.astype(np.int16)
    
    
    # Age
    x['age'] = pd.to_numeric(x.age,errors='coerce')
    
    # Null ages that exist
    nc = x.loc[x.age.isnull(),'ncodpers'].unique()
    nc2 = x[x.ncodpers.isin(nc) & x.age.notnull()].ncodpers.unique()
    age = x.loc[x.ncodpers.isin(nc2),['ncodpers','fecha_dato','age']]
    age['dummy'] = 100
    age.loc[age.age.notnull(),'dummy'] = age.age
    age = age.sort_values(['ncodpers','fecha_dato'],ascending=[1,0]).groupby('ncodpers',as_index=False).cummin().sort_values(['ncodpers','fecha_dato'])
    x.loc[age.index,'age'] = age.dummy
    
    # Rest are null
    x['age'] = x.age.fillna(-999).astype(np.int16)
    
    
    # Antiguedad
    x['antiguedad'] = pd.to_numeric(x.antiguedad,errors='coerce').fillna(-999).astype(np.int16)
    x.loc[x.antiguedad<-100,'antiguedad'] = -999
    ant = x[['ncodpers','fecha_dato','antiguedad']].copy()
    ant['real'] = 1
    ant['real2'] = ant.groupby('ncodpers')['real'].cumsum()-1
    tmp = ant['real2'].values.tolist()
    ant.sort_values(['ncodpers','fecha_dato'],ascending=[1,0],inplace=True)
    ant['tmp'] = tmp
    antmax = pd.DataFrame(ant.groupby('ncodpers')['antiguedad'].max())
    antmax.columns = ['antmax']
    ant = ant.merge(antmax,left_on='ncodpers',right_index=True,how='left')
    ant['antiguedad_real'] = ant.antmax-ant.tmp
    ant = ant.sort_values(['ncodpers','fecha_dato'])[['antiguedad_real']]
    x['antiguedad'] = ant['antiguedad_real']
    
    # Make everything antiguedad to start from 0
    nc = x.loc[(x.antiguedad<0) & (x.antiguedad>-99)].ncodpers.unique()
    bad = x.loc[x.ncodpers.isin(nc),['ncodpers','antiguedad']]
    bad = -bad.groupby('ncodpers').min()
    bad = x.loc[x.ncodpers.isin(nc),['ncodpers','antiguedad']].merge(bad,left_on='ncodpers',right_index=True,how='left')
    bad['antiguedad'] = bad['antiguedad_x'] + bad['antiguedad_y']
    x.loc[x.ncodpers.isin(nc),'antiguedad'] = bad['antiguedad']
    x.loc[x.antiguedad<-100,'antiguedad'] = -999
    x['antiguedad'] = x.antiguedad.astype(np.int16)
    
    
    # Renta
    x['renta'] = pd.to_numeric(x.renta,errors='coerce').astype(np.float32)
    
    # Fill those that have existing renta
    nc = x[x.renta.isnull()].ncodpers.unique()
    good = x.loc[(x.ncodpers.isin(nc)) & (x.renta.notnull())].ncodpers.unique()
    x.loc[x.ncodpers.isin(good)].groupby('ncodpers')['renta'].first()
    renta = x.loc[x.ncodpers.isin(good)].groupby('ncodpers')[['renta']].first()
    renta = x.loc[x.ncodpers.isin(good),['ncodpers']].merge(renta,left_on='ncodpers',right_index=True,how='left')
    x.loc[x.ncodpers.isin(good),'renta'] = renta.renta
    
    # Rest is unknown
    x['renta'] = x['renta'].fillna(-999).astype(np.float32)
    
    
    # tipodom
    x.drop('tipodom',axis=1,inplace=True)
    
    
    # ind_nuevo
    x.loc[x.ind_nuevo.isnull(),'ind_nuevo'] = 1
    
    
    # indrel_1mes
    x.loc[x.indrel_1mes=='P','indrel_1mes'] = 5
    x['indrel_1mes'] = pd.to_numeric(x.indrel_1mes,errors='coerce')
    
    
    # Canal entrada before encoding to integers
    x['ce_startletter'] = x.canal_entrada.str[0]
    x['ce_middleletter'] =  x.canal_entrada.str[1]
    x['ce_endletter'] = x.canal_entrada.str[2]
    
    # Categorical encoding
    for col in ['canal_entrada','cod_prov','conyuemp','ind_actividad_cliente','ind_empleado',
                'indext','indfall','indrel','indrel_1mes','indresi',
                'nomprov','pais_residencia','segmento','sexo','tiprel_1mes',
                'ind_nuevo','ce_startletter','ce_middleletter','ce_endletter']:
        print(col)
        print(x[col].isnull().sum())
        valc = x[col].value_counts()
        d = dict((k,v) for k,v in zip(valc.index,rankdata(valc,method='ordinal')))
        x[col] = x[col].map(d).fillna(-999).astype(np.int16)
        print(x[col].isnull().sum())
    
    
    # fecha_alta
    print('fecha_alta')
    x['fecha_alta'] = pd.to_datetime(x.fecha_alta,errors='coerce').astype(np.int64) # 27k
    x.loc[x.fecha_alta < 0,'fecha_alta'] = np.nan
    fecha_alta = x.groupby('ncodpers')[['fecha_alta']].first()
    fecha_alta = fecha_alta.interpolate()
    fecha_alta = x.loc[x.fecha_alta.isnull(),['ncodpers','fecha_alta']].merge(fecha_alta,left_on='ncodpers',right_index=True,how='inner')
    fecha_alta = fecha_alta.reindex(x.loc[x.fecha_alta.isnull()].index)
    x.loc[x.fecha_alta.isnull(),'fecha_alta'] = fecha_alta['fecha_alta_y']
    x['fecha_alta'] = pd.to_datetime(x.fecha_alta,errors='coerce')
    
    
    # fecha_dato
    x['fecha_dato'] = pd.to_datetime(x.fecha_dato,errors='coerce')
    
    x.loc[(x.fecha_dato - x.fecha_alta).dt.days < -100,'fecha_alta'] = np.nan
    x['fecha_alta'] = x.fecha_alta.dt.date
    x['fecha_alta'] = pd.to_datetime(x.fecha_alta,errors='coerce')
    
    
    # ult_fec_cli_1t
    print('ult_fec_cli_1t')
    x['ult_fec_cli_1t'] = pd.to_datetime(x.ult_fec_cli_1t,format='%Y-%m-%d',errors='coerce')
    x['days_since_primary'] = (x.ult_fec_cli_1t - x.fecha_dato).dt.days
    x.loc[x.days_since_primary.isnull(),'days_since_primary'] = -999
    x['days_since_primary'] = x.days_since_primary.astype(np.int16)
    x['ult_fec_cli_1t'] = x.ult_fec_cli_1t.astype(str)
    x.loc[x.ult_fec_cli_1t=='NaT','ult_fec_cli_1t'] = '0'
    x.loc[x.ult_fec_cli_1t!='NaT','ult_fec_cli_1t'] = '1'
    x['ult_fec_cli_1t'] = pd.to_numeric(x.ult_fec_cli_1t,errors='coerce')
    x['ult_fec_cli_1t'] = x.ult_fec_cli_1t.fillna(-999).astype(np.int16)
    
    
    # Products
    print('products')
    # Missing products (maybe didn't exist back then, all <= 2015-06-28
    x.loc[(x.fecha_dato<'2016-06-28') & (x.ind_nomina_ult1.isnull()) & (x.ind_cno_fin_ult1==1),['ind_nomina_ult1','ind_nom_pens_ult1']] = -999
    x.loc[(x.fecha_dato<'2016-06-28') & (x.ind_nomina_ult1.isnull()) & (x.ind_cno_fin_ult1==0),['ind_nomina_ult1','ind_nom_pens_ult1']] = -999
    
    # Test set products
    x.loc[x[prod_features()].isnull().any(1),prod_features()] = -999
    x[prod_features()] = x[prod_features()].fillna(-999).astype(np.int16)
    
    # Previous products to current line
    x.sort_values(['ncodpers','fecha_dato'],inplace=True)
    msk = x.ncodpers != x.ncodpers.shift(1)
    x[prod_features()] = x[prod_features()].shift(1)
    x.loc[msk,prod_features()] = np.nan
    x[prod_features()] = x[prod_features()].fillna(-999).astype(np.int16)
    
    
    ### ADDITIONAL FEATURES ###
    
    # Contract length
    x['contract_length'] = (x.fecha_dato - x.fecha_alta).dt.days
    
    # Contract % of age
    x['ratio_length_age'] = x.contract_length/x.age/365
    x.loc[x.age==-999,'ratio_length_age'] = np.nan
    x.loc[x.contract_length.isnull(),'ratio_length_age'] = np.nan
    
    # Latest contract % of first contract
    x['ratio_antiguedad_length'] = ( (x.contract_length - (x.antiguedad.astype(np.int32)*30.5)) / (x.contract_length) )
    x.loc[x.contract_length==0,'ratio_antiguedad_length'] = 1
    
    # Fillna
    x['contract_length'] = x.contract_length.fillna(-999).astype(np.int16)
    x['ratio_length_age'] = x.ratio_length_age.fillna(-999).astype(np.float16)
    x['ratio_antiguedad_length'] = x.ratio_antiguedad_length.fillna(-999).astype(np.float16)
    
    
    # Date features
    x['join_month'] = x.fecha_alta.dt.month.fillna(-999).astype(np.int16)
    x['join_year'] = x.fecha_alta.dt.year.fillna(-999).astype(np.int16)
    x['fetch_month'] = x.fecha_dato.dt.month.fillna(-999).astype(np.int16)
    x['fetch_year'] = x.fecha_dato.dt.year.fillna(-999).astype(np.int16)
    x['diff_fetch_join_month'] = np.abs(x.fetch_month - x.join_month)
    x.loc[x.diff_fetch_join_month>12,'diff_fetch_join_month'] = -999
    x['diff_fetch_join_month'] = x.diff_fetch_join_month.fillna(-999).astype(np.int16)
    
    
    # Products
    # Change in products
    change_cols = [name+"_change" for name in prod_features()]
    x[change_cols] = x[prod_features()] - x[prod_features()].shift(1)
    x.loc[msk,change_cols] = np.nan
    x.loc[(x[change_cols]>1).any(1),change_cols] = -999
    x[change_cols] = x[change_cols].fillna(-999).astype(np.int16)
    x[change_cols] = x[change_cols].replace(-1000,-999)
    
    # Number of products
    x['num_products'] = (x[prod_features()]==1).sum(1)
    
    # Number of products added
    x['num_products_added'] = (x[change_cols]==1).sum(1).astype(np.int8)
    x['num_products_removed'] = (x[change_cols]==-1).sum(1).astype(np.int8)
    
    # Days since last add / remove
    x['reset'] = (x.num_products_added>0).astype(np.int8)
    x.loc[msk,'reset'] = 1
    x['cumsum'] = x['reset'].cumsum()
    x['val'] = 1
    x['months_since_last_add'] = x.groupby(['cumsum'])['val'].cumsum()
    x.drop(['reset','cumsum','val'],axis=1,inplace=True)
    
    x['reset'] = (x.num_products_removed>0).astype(np.int8)
    x.loc[msk,'reset'] = 1
    x['cumsum'] = x['reset'].cumsum()
    x['val'] = 1
    x['months_since_last_remove'] = x.groupby(['cumsum'])['val'].cumsum()
    x.drop(['reset','cumsum','val'],axis=1,inplace=True)
    
    
    # Median renta for nomprov / exclude -999
    median_renta = x[x.renta!=-999].groupby(['cod_prov','fecha_dato','segmento'])['renta'].median().reset_index()
    median_renta.columns = ['cod_prov','fecha_dato','segmento','median_renta']
    x['median_renta'] = x[['cod_prov','fecha_dato','segmento']].merge(median_renta,on=['cod_prov','fecha_dato','segmento'],how='left')['median_renta']
    median_renta = x[x.renta!=-999].groupby(['fecha_dato'])['renta'].median().reset_index()
    median_renta.columns = ['fecha_dato','median_renta']
    median_renta = x[['fecha_dato']].merge(median_renta,on=['fecha_dato'],how='left')
    x.loc[x.median_renta.isnull(),'median_renta'] = median_renta[x.median_renta.isnull()]
    x['ratio_renta'] = (x.renta / x.median_renta)
    x.loc[x.renta<0,'ratio_renta'] = -999
    x['ratio_renta'] = x.ratio_renta.astype(np.float16)
    
    # Marriage index
    x['marriage_age'] = 28
    x['modifier'] = 0
    x.loc[x.sexo==2,'modifier'] += -2
    x.loc[x.renta<=101850,'modifier'] += -1
    x['marriage_age_mod'] = x.marriage_age + x.modifier
    x['marriage_index'] = (x.age <= x.marriage_age_mod).astype(np.int16)
    x.loc[x.renta<0,'marriage_index'] = -999
    x.drop(['marriage_age','modifier','marriage_age_mod'],axis=1,inplace=True)
    
    
    # Birthday next month
    x['birthday_next_month'] = (x.age != x.age.shift(-1)).astype(np.int16)
    msk = x.ncodpers != x.ncodpers.shift(-1)
    x.loc[msk,'birthday_next_month'] = 0
    x.loc[x.age==-999,'birthday_next_month'] = -999
    
    # Bimodal -> Less or greater than 32 yrs old?
    x['age_group'] = (x.age>=32).astype(np.int16)
    x.loc[x.age==-999,'age_group'] = -999
    
    
    # Categorical Differences
    msk = x.ncodpers != x.ncodpers.shift(1)
    for col in ['canal_entrada','cod_prov','conyuemp','ind_actividad_cliente','ind_empleado',
                'indext','indfall','indrel','indrel_1mes','indresi',
                'nomprov','pais_residencia','segmento','sexo','tiprel_1mes',
                'ind_nuevo']:
        name = col+"_diff"
        print(name)
        x[name] = x[col] - x[col].shift(1)
        x.loc[msk,name] = 0
        x[name] = x[name].fillna(-999).astype(np.int16)
        
    # Product sums
    x['num_junior'] = x['ind_cder_fin_ult1'] + x['ind_tjcr_fin_ult1']
    x['num_random'] = x['ind_ecue_fin_ult1'] + x['ind_cco_fin_ult1']
    x['num_pension2'] = x['ind_nom_pens_ult1'] + x['ind_nomina_ult1'] + x['ind_recibo_ult1']
    x['num_basic'] = x['ind_nom_pens_ult1'] + x['ind_nomina_ult1'] + x['ind_cno_fin_ult1']
    x['num_rare'] = x['ind_ahor_fin_ult1'] + x['ind_aval_fin_ult1']
    x['num_particular'] = x['ind_ctma_fin_ult1'] + x['ind_ctop_fin_ult1'] + x['ind_ctpp_fin_ult1']
    x['num_deposits'] = x['ind_deco_fin_ult1'] + x['ind_deme_fin_ult1'] + x['ind_dela_fin_ult1']
    x['num_pension'] =  x['ind_nomina_ult1'] + x['ind_nom_pens_ult1']
    x['num_finance'] = x['ind_fond_fin_ult1'] + x['ind_valo_fin_ult1'] + x['ind_cder_fin_ult1'] + x['ind_plan_fin_ult1']
    x['num_credit'] = x['ind_recibo_ult1'] + x['ind_tjcr_fin_ult1']
    x['num_home'] =  x['ind_viv_fin_ult1'] + x['ind_hip_fin_ult1'] + x['ind_pres_fin_ult1']
    x['num_work'] = x['ind_cno_fin_ult1'] + x['ind_reca_fin_ult1']
    x['num_modern'] = x['ind_ecue_fin_ult1'] + x['ind_ctju_fin_ult1']
    cols = ['num_rare','num_particular','num_deposits','num_pension','num_finance','num_credit','num_home','num_work','num_modern','num_junior',
            'num_random','num_pension2','num_basic']
    x.loc[(x[cols]<0).any(1),cols] = -999
    x[cols] = x[cols].fillna(-999).astype(np.int16)
    
    
    # Belongs to two of the most common indicators
    for col in ['cod_prov','segmento','indrel_1mes','tiprel_1mes','canal_entrada']:
        name = col+'_most_common'
        x[name] = (x[col] >= (x[col].max())).fillna(-999).astype(np.int16)
        
        
    # Number of accounts per nomprov per fecha_dato
    # Median salary for nomprov per fecha_dato
    # Median age for nomprov per fecha_dato
    g = x.groupby(['nomprov','fecha_dato'])['ncodpers','renta','age'] \
        .aggregate({'nunique':lambda x: x.nunique(), 'median':lambda x: x.median()}) \
        .reset_index()
    g.columns = ['nomprov','fecha_dato'] + ['_'.join(col).strip() for col in g.columns.values[2:]]
    g = g[['nomprov','fecha_dato','nunique_ncodpers','median_renta','median_age']]
    x = x.merge(g,on=['nomprov','fecha_dato'],how='left')
    print(x)
    x['ratio_age'] = x.age/x.median_age
    x['ratio_age'] = x.ratio_age.astype(np.float32)
    x['nunique_ncodpers'] = x.nunique_ncodpers.astype(np.int32)
    
    x = x.fillna(-999)
    
    endshape = x.shape[0]
    print('End shape',endshape)
    
    assert startshape==endshape
    
    return(x)
    
# Preprocess function 1
def preprocess(x):
    
    startshape = x.shape[0]
    print('Start shape',startshape)
    
    # Take original NaN count for every row
    print('n_nan')
    x['n_nan'] = x.iloc[:,:24].isnull().sum(1)
    x['n_nan'] = x.n_nan.astype(np.int16)
    
    # age
    print('age')
    x['age'] = pd.to_numeric(x.age,errors='coerce') #27k
    age = x.groupby('ncodpers')[['age']].mean().interpolate()
    age = x.loc[x.age.isnull(),['ncodpers','age']].merge(age,left_on='ncodpers',right_index=True,how='inner').sort_index()
    x.loc[x.age.isnull(),'age'] = age['age_y']
    x['age'] = x.age.astype(np.int16)
    
    age = (x.groupby('ncodpers')['age'].nunique() > 2)
    nc = age.index[age]
    age = x.loc[x.ncodpers.isin(nc)].groupby('ncodpers')[['age']].median()
    age = x.loc[x.ncodpers.isin(nc),['ncodpers','fecha_dato','age']].merge(age,left_on='ncodpers',right_index=True,how='left')
    age['real_age'] = age.age_x - age.age_y
    age.loc[age.real_age<0,'real_age'] = 0
    age.loc[age.real_age>1,'real_age'] = 1
    age['real_age'] = age.real_age.astype(np.int16)
    age['real_age'] = age.real_age + age.age_y
    age['real_age'] = age.real_age.astype(np.int16)
    age['real_age'] = age.sort_values(['ncodpers','fecha_dato']).groupby('ncodpers')['real_age'].cummax()
    age = age.reindex(x.loc[x.ncodpers.isin(nc)].index)
    x.loc[x.ncodpers.isin(nc),'age'] = age['real_age']
    x['age'] = x.age.astype(np.int16)
    
    # ind_empleado
    print('ind_empleado')
    x.loc[x.ind_empleado.isnull(),'ind_empleado'] = 'N' #27k
    d_ind_empleado = dict((k,v) for k,v in zip(x.ind_empleado.value_counts().index,rankdata(x.ind_empleado.value_counts(),method='ordinal')))
    x['ind_empleado'] = x.ind_empleado.map(d_ind_empleado)
    x['ind_empleado'] = x.ind_empleado.astype(np.int16)
    
    # pais_residencia
    print('pais_residencia')
    x.loc[x.pais_residencia.isnull(),'pais_residencia'] = 'ES' #27k
    d_pais_residencia = dict((k,v) for k,v in zip(x.pais_residencia.value_counts().index,rankdata(x.pais_residencia.value_counts(),method='ordinal')))
    x['pais_residencia'] = x.pais_residencia.map(d_pais_residencia)
    x['pais_residencia'] = x.pais_residencia.astype(np.int16)
    
    # sexo
    print('sexo')
    bad = x.loc[x.sexo.isnull()].ncodpers.unique() # Those that we know sexo for
    known = x.loc[x.ncodpers.isin(bad) & x.sexo.notnull()].groupby('ncodpers')[['sexo']].first()
    known_nc = known.index.unique()
    known = x.loc[x.ncodpers.isin(known_nc),['ncodpers','sexo']].merge(known,left_on='ncodpers',right_index=True,how='inner')
    known = known.reindex(x.loc[x.ncodpers.isin(known_nc)].index)
    x.loc[x.ncodpers.isin(known_nc),'sexo'] = known['sexo_y']
    bad = x.loc[x.sexo.isnull()].ncodpers.unique()
    np.random.seed(2016)
    H = np.random.choice(bad,size=int(np.ceil(len(bad)/2)),replace=False)
    V = set(bad).difference(H)
    x.loc[x.ncodpers.isin(H),'sexo'] = 'H'
    x.loc[x.ncodpers.isin(V),'sexo'] = 'V'
    d_sexo = dict((k,v) for k,v in zip(x.sexo.value_counts().index,rankdata(x.sexo.value_counts(),method='ordinal')))
    x['sexo'] = x.sexo.map(d_sexo)
    x['sexo'] = x.sexo.astype(np.int16)
    
    # fecha_alta
    print('fecha_alta')
    x['fecha_alta'] = pd.to_datetime(x.fecha_alta,errors='coerce').astype(np.int64) # 27k
    x.loc[x.fecha_alta < 0,'fecha_alta'] = np.nan
    fecha_alta = x.groupby('ncodpers')[['fecha_alta']].first()
    fecha_alta = fecha_alta.interpolate()
    fecha_alta = x.loc[x.fecha_alta.isnull(),['ncodpers','fecha_alta']].merge(fecha_alta,left_on='ncodpers',right_index=True,how='inner')
    fecha_alta = fecha_alta.reindex(x.loc[x.fecha_alta.isnull()].index)
    x.loc[x.fecha_alta.isnull(),'fecha_alta'] = fecha_alta['fecha_alta_y']
    x['fecha_alta'] = pd.to_datetime(x.fecha_alta,errors='coerce')
    
    # antiguedad
    print('antiguedad')
    x['antiguedad_startswith_blank'] = 0
    x.loc[x.antiguedad.notnull(),'antiguedad_startswith_blank'] = x.loc[x.antiguedad.notnull(),'antiguedad'].str.startswith(" ")
    x.loc[x.antiguedad_startswith_blank.isnull(),'antiguedad_startswith_blank'] = False
    x['antiguedad_startswith_blank'] = x.antiguedad_startswith_blank.astype(np.int16)
    x['antiguedad'] = pd.to_numeric(x.antiguedad,errors='coerce')
    x.loc[x.antiguedad <= 0,'antiguedad'] = np.nan # 27k + these
    x['fecha_dato'] = pd.to_datetime(x.fecha_dato,errors='coerce')
    x.loc[x.antiguedad.isnull(),'antiguedad'] = (x.loc[x.antiguedad.isnull(),'fecha_dato'].dt.year - x.loc[x.antiguedad.isnull(),'fecha_alta'].dt.year) * 12 + \
                                                 (x.loc[x.antiguedad.isnull(),'fecha_dato'].dt.month - x.loc[x.antiguedad.isnull(),'fecha_alta'].dt.month)
    rl = pd.DataFrame(x[['ncodpers','fecha_dato','antiguedad']].sort_values(['ncodpers','fecha_dato'],ascending=[1,0]).groupby('ncodpers')[['antiguedad']].cumcount(ascending=True))
    rl = rl.merge(x[['ncodpers']],left_index=True,right_index=True,how='left')
    rlmax = x.groupby('ncodpers')[['antiguedad']].max()
    rl = rl.merge(rlmax,left_on='ncodpers',right_index=True,how='left')
    rl.columns = ['subtr','ncodpers','antiguedad']
    rl['antiguedad'] = rl.antiguedad - rl.subtr
    rl = rl.sort_index()
    x['antiguedad'] = rl['antiguedad']
    x['antiguedad'] = x.antiguedad.astype(np.int16)
    
    x.loc[x.age*12 < x.antiguedad,'age'] = (x.loc[x.age*12 < x.antiguedad,'antiguedad']/12).astype(np.int16)

    # ind_nuevo
    print('ind_nuevo')
    x.loc[(x.ind_nuevo.isnull()) & (x.antiguedad <= 6),'ind_nuevo'] = 1
    x.loc[(x.ind_nuevo.isnull()) & (x.antiguedad > 6),'ind_nuevo'] = 0
    x['ind_nuevo'] = x.ind_nuevo.astype(np.int16)
    
    # indrel
    print('indrel')
    x.loc[x.indrel.isnull(),'indrel'] = 1
    x['indrel'] = x.indrel.astype(np.int16)
    
    # ult_fec_cli_1t
    print('ult_fec_cli_1t')
    x['ult_fec_cli_1t'] = pd.to_datetime(x.ult_fec_cli_1t,format='%Y-%m-%d',errors='coerce')
    x['primary_days_left'] = (x.ult_fec_cli_1t - x.fecha_dato).dt.days + 31
    x.loc[x.primary_days_left.isnull(),'primary_days_left'] = 0
    x['primary_days_left'] = x.primary_days_left.astype(np.int16)
    x['ult_fec_cli_1t'] = x.ult_fec_cli_1t.astype(str)
    x.loc[x.ult_fec_cli_1t=='NaT','ult_fec_cli_1t'] = '0'
    x.loc[x.ult_fec_cli_1t!='NaT','ult_fec_cli_1t'] = '1'
    x['ult_fec_cli_1t'] = pd.to_numeric(x.ult_fec_cli_1t,errors='coerce')
    x['ult_fec_cli_1t'] = x.ult_fec_cli_1t.astype(np.int16)
    
    # indrel_1mes
    print('indrel_1mes')
    x.loc[x.indrel_1mes=='P','indrel_1mes'] = 5
    x['indrel_1mes'] = pd.to_numeric(x.indrel_1mes,errors='coerce')
    x.loc[x.indrel_1mes.isnull(),'indrel_1mes'] = 6
    x['indrel_1mes'] = x.indrel_1mes.astype(np.int16)
    d_indrel_1mes = dict((k,v) for k,v in zip(x.indrel_1mes.value_counts().index,rankdata(x.indrel_1mes.value_counts(),method='ordinal')))
    x['indrel_1mes'] = x.indrel_1mes.map(d_indrel_1mes)
    x['indrel_1mes'] = x.indrel_1mes.astype(np.int16)
    
    # tiprel_1mes
    print('tiprel_1mes')
    x.loc[x.tiprel_1mes.isnull(),'tiprel_1mes'] = 'UNKNOWN'
    d_tiprel_1mes = dict((k,v) for k,v in zip(x.tiprel_1mes.value_counts().index,rankdata(x.tiprel_1mes.value_counts(),method='ordinal')))
    x['tiprel_1mes'] = x.tiprel_1mes.map(d_tiprel_1mes)
    x['tiprel_1mes'] = x.tiprel_1mes.astype(np.int16)
    
    # indresi
    print('indresi')
    x.loc[x.indresi.isnull(),'indresi'] = 'S' #27k
    x.loc[x.indresi=='S','indresi'] = 1
    x.loc[x.indresi=='N','indresi'] = 0
    x['indresi'] = x.indresi.astype(np.int16)
    
    # indext
    print('indext')
    x.loc[x.indext.isnull(),'indext'] = 'N'
    x.loc[x.indext=='N','indext'] = 0
    x.loc[x.indext=='S','indext'] = 1
    x['indext'] = x.indext.astype(np.int16)
    
    # conyuemp
    print('conyuemp')
    x.loc[x.conyuemp.isnull(),'conyuemp'] = 'UNKNOWN'
    d_conyuemp = dict((k,v) for k,v in zip(x.conyuemp.value_counts().index,rankdata(x.conyuemp.value_counts(),method='ordinal')))
    x['conyuemp'] = x.conyuemp.map(d_conyuemp).astype(np.int16)
    
    # canal_entrada
    print('canal_entrada')
    bad = x.loc[x.canal_entrada.isnull(),'ncodpers'].unique()
    canal_entrada = x.loc[x.ncodpers.isin(bad)].groupby('ncodpers')[['canal_entrada']].first()
    canal_entrada = x.loc[x.canal_entrada.isnull(),['ncodpers','canal_entrada']].merge(canal_entrada,left_on='ncodpers',right_index=True,how='right')
    canal_entrada = canal_entrada.reindex(x.loc[x.canal_entrada.isnull()].index)
    x.loc[x.canal_entrada.isnull(),'canal_entrada'] = canal_entrada['canal_entrada_y']
    
    canal_entrada = x[['age','canal_entrada']]
    canal_entrada['bins'] = pd.cut(x.age,[0,10,20,30,40,50,60,70,80,90,100,200])
    top_age = pd.DataFrame(canal_entrada.groupby(['bins'])['canal_entrada'].value_counts().groupby(level=0).nlargest(1))
    top_age.columns = ['count']
    top_age.reset_index(level=0,inplace=True)
    top_age.reset_index(level=1,inplace=True)
    top_age = top_age[['bins','canal_entrada']].reset_index(drop=True)
    canal_entrada = canal_entrada.merge(top_age,on='bins')
    canal_entrada = canal_entrada.reindex(canal_entrada.index[x.loc[x.canal_entrada.isnull()].index])
    canal_entrada = canal_entrada.loc[x.loc[x.canal_entrada.isnull()].index,['canal_entrada_y']]
    x.loc[x.canal_entrada.isnull(),'canal_entrada'] = canal_entrada['canal_entrada_y']
    
    x['ce_startletter'] = x.canal_entrada.str[0]
    d_ce_startletter = dict((k,v) for k,v in zip(x.ce_startletter.value_counts().index,rankdata(x.ce_startletter.value_counts(),method='ordinal')))
    x['ce_startletter'] = x.ce_startletter.map(d_ce_startletter).astype(np.int16)
    x['ce_middleletter'] =  x.canal_entrada.str[1]
    d_ce_middleletter = dict((k,v) for k,v in zip(x.ce_middleletter.value_counts().index,rankdata(x.ce_middleletter.value_counts(),method='ordinal')))
    x['ce_middleletter'] = x.ce_middleletter.map(d_ce_middleletter).astype(np.int16)
    x['ce_endletter'] = x.canal_entrada.str[2]
    d_ce_endletter = dict((k,v) for k,v in zip(x.ce_endletter.value_counts().index,rankdata(x.ce_endletter.value_counts(),method='ordinal')))
    x['ce_endletter'] = x.ce_endletter.map(d_ce_endletter).astype(np.int16)
    
    d_canal_entrada = dict((k,v) for k,v in zip(x.canal_entrada.value_counts().index,rankdata(x.canal_entrada.value_counts(),method='ordinal')))
    x['canal_entrada'] = x.canal_entrada.map(d_canal_entrada).astype(np.int16)
    
    # indfall
    print('indfall')
    x.loc[x.indfall.isnull(),'indfall'] = 'N' #27k
    x.loc[x.indfall=='N','indfall'] = 0
    x.loc[x.indfall=='S','indfall'] = 1
    x['indfall'] = x.indfall.astype(np.int16)
    
    # tipodom
    print('tipodom')
    x.drop('tipodom',axis=1,inplace=True)
    
    # cod_prov, nomprov
    print('cod_prov,nomprov')
    x['cod_prov'] = pd.to_numeric(x.cod_prov,errors='coerce')
    bad = x.loc[x.cod_prov.isnull(),'ncodpers'].unique()
    cod_prov = x.loc[x.ncodpers.isin(bad)].groupby('ncodpers')[['cod_prov']].first()
    cod_prov = cod_prov[cod_prov.notnull().values]
    cod_prov = x.loc[x.cod_prov.isnull(),['ncodpers','cod_prov']].merge(cod_prov,left_on='ncodpers',right_index=True,how='inner')
    x.loc[cod_prov.index,'cod_prov'] = cod_prov['cod_prov_y']
    
    top_res = pd.DataFrame(x.groupby(['pais_residencia'])['cod_prov'].value_counts().groupby(level=0).nlargest(1)).reset_index(level=1)
    top_res.columns = ['pais_residencia','count']
    top_res = top_res.reset_index(level=1)[['pais_residencia','cod_prov']].reset_index(drop=True)
    nomprov = pd.DataFrame(x.groupby('cod_prov')['nomprov'].first())
    top_res = top_res.merge(nomprov,left_on='cod_prov',right_index=True,how='left')
    top_res = x.loc[x.cod_prov.isnull(),['ncodpers','pais_residencia','nomprov']].reset_index().merge(top_res,left_on='pais_residencia',right_on='pais_residencia',how='left').set_index('index')
    tmp_d = dict((v,k+53) for k,v in enumerate(top_res.loc[top_res.cod_prov.isnull()].pais_residencia.unique()))
    top_res.loc[top_res.cod_prov.isnull(),'cod_prov'] = top_res.loc[top_res.cod_prov.isnull(),'pais_residencia'].map(tmp_d)
    x.loc[x.cod_prov.isnull(),'nomprov'] = top_res['nomprov_y']
    x.loc[x.cod_prov.isnull(),'cod_prov'] = top_res['cod_prov']
    d_cod_prov = dict((k,v) for k,v in zip(x.cod_prov.value_counts().index,rankdata(x.cod_prov.value_counts(),method='ordinal')))
    x['cod_prov'] = x.cod_prov.map(d_cod_prov).astype(np.int16)
    
    nomprov = pd.DataFrame(x.groupby('cod_prov')['nomprov'].first()).dropna()
    nomprov = x.loc[x.nomprov.isnull(),['cod_prov']].reset_index().merge(nomprov,left_on='cod_prov',right_index=True,how='left').set_index('index')
    x.loc[x.nomprov.isnull(),'nomprov'] = nomprov['nomprov']
    x.loc[x.nomprov.isnull(),'nomprov'] = x.loc[x.nomprov.isnull(),'pais_residencia']
    
    
    # ind_actividad_cliente
    print('ind_actividad_cliente')
    x.loc[x.ind_actividad_cliente.isnull(),'ind_actividad_cliente'] = 0 #27k, no idea
    x['ind_actividad_cliente'] = x.ind_actividad_cliente.astype(np.int16)
    
    # segmento
    print('segmento')
    bad = x[x.segmento.isnull()].ncodpers.unique()
    segmento = x[x.ncodpers.isin(bad)].groupby('ncodpers')[['segmento']].first().dropna()
    segmento = x.loc[x.segmento.isnull(),['ncodpers','segmento']].merge(segmento,left_on='ncodpers',right_index=True,how='left')
    x.loc[x.segmento.isnull(),'segmento'] = segmento['segmento_y']
    x.loc[x.segmento.isnull(),'segmento'] = 'UNKNOWN'
    d_segmento = dict((k,v) for k,v in zip(x.segmento.value_counts().index,rankdata(x.segmento.value_counts(),method='ordinal')))
    x['segmento'] = x.segmento.map(d_segmento).astype(np.int16)
    
    # renta
    print('renta')
    x['renta'] = pd.to_numeric(x.renta,errors='coerce')
    renta = x[['ncodpers','nomprov','age','renta']]
    renta['age'] = pd.cut(renta.age, [0,20,30,40,60,80,200])
    xf = renta
    renta = renta.groupby(['nomprov','age'])['renta'] \
                .aggregate({'median':lambda x: x.median(),'nunique':lambda x: x.nunique()}) \
                .reset_index() \
                .dropna()
    renta = renta[renta.nunique >= 50]
    renta = xf.loc[xf.renta.isnull(),['ncodpers','nomprov','age']].reset_index().merge(renta,on=['nomprov','age'],how='left').set_index('index')
    x.loc[x.renta.isnull(),'renta'] = renta['median']
    
    renta = x.groupby(['nomprov'])['renta'].aggregate({'median':lambda x: x.median(), 'nunique':lambda x: x.nunique()}).reset_index()
    renta = x.loc[x.renta.isnull(),['nomprov']].reset_index().merge(renta,on=['nomprov'],how='left').set_index('index')
    x.loc[x.renta.isnull(),'renta'] = renta['median']
    
    x.loc[x.renta.isnull(),'renta'] = x.renta.median()
    x['renta'] = x.renta.astype(np.float32)
    
    # Additional features
    print('Additional features')
    # Number of accounts per nomprov per fecha_dato
    # Median salary for nomprov per fecha_dato
    # Median age for nomprov per fecha_dato
    g = x.groupby(['nomprov','fecha_dato'])['ncodpers','renta','age'] \
        .aggregate({'nunique':lambda x: x.nunique(), 'median':lambda x: x.median()}) \
        .reset_index()
    g.columns = ['nomprov','fecha_dato'] + ['_'.join(col).strip() for col in g.columns.values[2:]]
    g = g[['nomprov','fecha_dato','nunique_ncodpers','median_renta','median_age']]
    x = x.merge(g,on=['nomprov','fecha_dato'],how='left')
    x['ratio_renta'] = x.renta/x.median_renta
    x['ratio_age'] = x.age/x.median_age
    x.drop(['median_renta','median_age'],axis=1,inplace=True)
    x['ratio_renta'] = x.ratio_renta.astype(np.float32)
    x['ratio_age'] = x.ratio_age.astype(np.float32)
    x['nunique_ncodpers'] = x.nunique_ncodpers.astype(np.int32)
    
    # Age cut
    print('age cut')
    age_bin = pd.cut(x.age,[0,20,30,40,60,80,100,300])
    x['age_bin'] = age_bin
    d_age_bin = dict((k,v) for k,v in zip(age_bin.value_counts().index,rankdata(age_bin.value_counts(),method='ordinal')))
    x['age_bin'] = x.age_bin.map(d_age_bin).astype(np.int16)
    
    # Contract length
    print('contract length')
    x['contract_length'] = (x.fecha_dato-x.fecha_alta).dt.days.astype(np.int16)
    x.loc[x.contract_length<0,'contract_length'] = 0
    
    # Year born
    print('year born')
    x['year_born'] = (x.fecha_dato - pd.to_timedelta(x.age.astype(np.int32)*365, unit='D')).dt.year.astype(np.int16)
    
    # Other that were previously in the csv dict version
    print('contract percentage, month')
    x['contract_percentage_age'] = (x.contract_length / (x.age.astype(np.int32)*365)).astype(np.float32)
    x['contract_percentage_antiguedad'] = ( (x.contract_length - (x.antiguedad.astype(np.int32)*12)) / (x.contract_length) ).astype(np.float32)
    x.loc[x.contract_percentage_antiguedad.isnull(),'contract_percentage_antiguedad'] = 0
    x['month'] = x.fecha_dato.dt.month
    
    # Products
    print('products')
    # Missing products (maybe didn't exist back then, all <= 2015-06-28
    x.loc[(x.fecha_dato<'2016-06-28') & (x.ind_nomina_ult1.isnull()) & (x.ind_cno_fin_ult1==1),['ind_nomina_ult1','ind_nom_pens_ult1']] = 1.0
    x.loc[(x.fecha_dato<'2016-06-28') & (x.ind_nomina_ult1.isnull()) & (x.ind_cno_fin_ult1==0),['ind_nomina_ult1','ind_nom_pens_ult1']] = 0.0
    
    # Test set products
    x.loc[x[prod_features()].isnull().any(1),prod_features()] = -999
    x[prod_features()] = x[prod_features()].astype(np.int16)
    
    # Previous products to current line
    print('products')
    x.sort_values(['ncodpers','fecha_dato'],inplace=True)
    msk = x.ncodpers != x.ncodpers.shift(1)
    #original = x.loc[msk,prod_features()]
    x[prod_features()] = x[prod_features()].shift(1)
    x.loc[msk,prod_features()] = np.nan#original
    x[prod_features()] = x[prod_features()].fillna(-999).astype(np.int16)
    
    # Add cumsum?
    
    # Products added since last
    print('products added or removed')
    change_cols = [name+"_change" for name in prod_features()]
    x[change_cols] = x[prod_features()] - x[prod_features()].shift(1)
    x.loc[msk,change_cols] = np.nan
    x.loc[(x[change_cols]>1).any(1),change_cols] = -999
    x[change_cols] = x[change_cols].fillna(-999).astype(np.int16)
    
    # Number of products added
    x['num_products_added'] = (x[change_cols]==1).sum(1)
    x['num_products_removed'] = (x[change_cols]==-1).sum(1)
    
    # Number of products in "product groups"
    print('product groups')
    x['num_rare'] = x['ind_ahor_fin_ult1'] + x['ind_aval_fin_ult1']
    x['num_particular'] = x['ind_ctma_fin_ult1'] + x['ind_ctop_fin_ult1'] + x['ind_ctpp_fin_ult1']
    x['num_deposits'] = x['ind_deco_fin_ult1'] + x['ind_deme_fin_ult1'] + x['ind_dela_fin_ult1']
    x['num_pension'] =  x['ind_nomina_ult1'] + x['ind_nom_pens_ult1']
    x['num_finance'] = x['ind_fond_fin_ult1'] + x['ind_valo_fin_ult1'] + x['ind_cder_fin_ult1'] + x['ind_plan_fin_ult1']
    x['num_credit'] = x['ind_recibo_ult1'] + x['ind_tjcr_fin_ult1']
    x['num_home'] =  x['ind_viv_fin_ult1'] + x['ind_hip_fin_ult1'] + x['ind_pres_fin_ult1']
    x['num_work'] = x['ind_cno_fin_ult1'] + x['ind_reca_fin_ult1']
    x['num_modern'] = x['ind_ecue_fin_ult1'] + x['ind_ctju_fin_ult1']
    cols = ['num_rare','num_particular','num_deposits','num_pension','num_finance',
            'num_credit','num_home','num_work','num_modern']
    x.loc[(x[cols]<0).any(1),cols] = -999
    x[cols] = x[cols].astype(np.int16)
    
    # Indicators for change
    print('change indicators')
    msk = x.ncodpers != x.ncodpers.shift(1)
    
    x['ind_actividad_cliente_diff'] = x.ind_actividad_cliente - x.ind_actividad_cliente.shift(1)
    x.loc[msk,'ind_actividad_cliente_diff'] = 0
    x['ind_actividad_cliente_diff'] = x.ind_actividad_cliente_diff.astype(np.int16)
    
    x['ind_nuevo_diff'] = x.ind_nuevo - x.ind_nuevo.shift(1)
    x.loc[msk,'ind_nuevo_diff'] = 0
    x['ind_nuevo_diff'] = x.ind_nuevo_diff.astype(np.int16)
    
    x['cod_prov_diff'] = x.cod_prov - x.cod_prov.shift(1)
    x.loc[msk,'cod_prov_diff'] = 0
    x['cod_prov_diff'] = x.cod_prov_diff.astype(np.int16)
    
    x['segmento_diff'] = x.segmento - x.segmento.shift(1)
    x.loc[msk,'segmento_diff'] = 0
    x['segmento_diff'] = x.segmento_diff.astype(np.int16)
    
    x['indrel_diff'] = x.indrel - x.indrel.shift(1)
    x.loc[msk,'indrel_diff'] = 0
    x['indrel_diff'] = x.indrel_diff.astype(np.int16)
    
    x['indrel_1mes_diff'] = x.indrel_1mes - x.indrel_1mes.shift(1)
    x.loc[msk,'indrel_1mes_diff'] = 0
    x['indrel_1mes_diff'] = x.indrel_1mes_diff.astype(np.int16)
    
    x['tiprel_1mes_diff'] = x.tiprel_1mes - x.tiprel_1mes.shift(1)
    x.loc[msk,'tiprel_1mes_diff'] = 0
    x['tiprel_1mes_diff'] = x.tiprel_1mes_diff.astype(np.int16)
    
    x['canal_entrada_diff'] = x.canal_entrada - x.canal_entrada.shift(1)
    x.loc[msk,'canal_entrada_diff'] = 0
    x['canal_entrada_diff'] = x.canal_entrada_diff.astype(np.int16)
    
    x['age_diff'] = x.age - x.age.shift(1)
    x.loc[msk,'age_diff'] = 0
    x['age_diff'] = x.age_diff.astype(np.int16)
    
    # Belongs to two of the most common indicators
    for col in ['cod_prov','segmento','indrel_1mes','tiprel_1mes','canal_entrada']:
        name = col+'_most_common'
        x[name] = (x[col] >= (x[col].max())).astype(np.int8)
    
    endshape = x.shape[0]
    print('End shape',endshape)
    
    assert startshape==endshape
    
    return(x)
    
# Preprocess function 2
def preprocess2(x):
    
    startshape = x.shape[0]
    print('Start shape',startshape)
    
    # Take original NaN count for every row
    print('n_nan')
    x['n_nan'] = x.iloc[:,:24].isnull().sum(1)
    x['n_nan'] = x.n_nan.astype(np.int16)
    
    # age
    print('age')
    x['age'] = pd.to_numeric(x.age,errors='coerce')
    x['age'] = x.age.fillna(-999).astype(np.int16)
    
    # ind_empleado
    print('ind_empleado')
    d_ind_empleado = dict((k,v) for k,v in zip(x.ind_empleado.value_counts().index,rankdata(x.ind_empleado.value_counts(),method='ordinal')))
    x['ind_empleado'] = x.ind_empleado.map(d_ind_empleado)
    x['ind_empleado'] = x.ind_empleado.fillna(-999).astype(np.int16)
    
    # pais_residencia
    print('pais_residencia')
    d_pais_residencia = dict((k,v) for k,v in zip(x.pais_residencia.value_counts().index,rankdata(x.pais_residencia.value_counts(),method='ordinal')))
    x['pais_residencia'] = x.pais_residencia.map(d_pais_residencia)
    x['pais_residencia'] = x.pais_residencia.fillna(-999).astype(np.int16)
    
    # sexo
    print('sexo')
    d_sexo = dict((k,v) for k,v in zip(x.sexo.value_counts().index,rankdata(x.sexo.value_counts(),method='ordinal')))
    x['sexo'] = x.sexo.map(d_sexo)
    x['sexo'] = x.sexo.fillna(-999).astype(np.int16)
    
    # fecha_alta
    print('fecha_alta')
    x['fecha_alta'] = pd.to_datetime(x.fecha_alta,errors='coerce').astype(np.int64) # 27k
    x.loc[x.fecha_alta < 0,'fecha_alta'] = np.nan
    fecha_alta = x.groupby('ncodpers')[['fecha_alta']].first()
    fecha_alta = fecha_alta.interpolate()
    fecha_alta = x.loc[x.fecha_alta.isnull(),['ncodpers','fecha_alta']].merge(fecha_alta,left_on='ncodpers',right_index=True,how='inner')
    fecha_alta = fecha_alta.reindex(x.loc[x.fecha_alta.isnull()].index)
    x.loc[x.fecha_alta.isnull(),'fecha_alta'] = fecha_alta['fecha_alta_y']
    x['fecha_alta'] = pd.to_datetime(x.fecha_alta,errors='coerce')
    
    # antiguedad
    print('antiguedad')
    x['antiguedad'] = pd.to_numeric(x.antiguedad,errors='coerce').fillna(-999).astype(np.int16)

    # ind_nuevo
    print('ind_nuevo')
    x['ind_nuevo'] = x.ind_nuevo.fillna(-999).astype(np.int16)
    
    # indrel
    print('indrel')
    x.loc[x.indrel.isnull(),'indrel'] = 1
    x['indrel'] = x.indrel.fillna(-999).astype(np.int16)
    
    # ult_fec_cli_1t
    print('ult_fec_cli_1t')
    x['ult_fec_cli_1t'] = pd.to_datetime(x.ult_fec_cli_1t,format='%Y-%m-%d',errors='coerce')
    x['primary_days_left'] = (x.ult_fec_cli_1t - pd.to_datetime(x.fecha_dato)).dt.days + 31
    x.loc[x.primary_days_left.isnull(),'primary_days_left'] = 0
    x['primary_days_left'] = x.primary_days_left.astype(np.int16)
    x['ult_fec_cli_1t'] = x.ult_fec_cli_1t.astype(str)
    x.loc[x.ult_fec_cli_1t=='NaT','ult_fec_cli_1t'] = '0'
    x.loc[x.ult_fec_cli_1t!='NaT','ult_fec_cli_1t'] = '1'
    x['ult_fec_cli_1t'] = pd.to_numeric(x.ult_fec_cli_1t,errors='coerce')
    x['ult_fec_cli_1t'] = x.ult_fec_cli_1t.fillna(-999).astype(np.int16)
    
    # indrel_1mes
    print('indrel_1mes')
    d_indrel_1mes = dict((k,v) for k,v in zip(x.indrel_1mes.value_counts().index,rankdata(x.indrel_1mes.value_counts(),method='ordinal')))
    x['indrel_1mes'] = x.indrel_1mes.map(d_indrel_1mes)
    x['indrel_1mes'] = x.indrel_1mes.fillna(-999).astype(np.int16)
    
    # tiprel_1mes
    print('tiprel_1mes')
    d_tiprel_1mes = dict((k,v) for k,v in zip(x.tiprel_1mes.value_counts().index,rankdata(x.tiprel_1mes.value_counts(),method='ordinal')))
    x['tiprel_1mes'] = x.tiprel_1mes.map(d_tiprel_1mes)
    x['tiprel_1mes'] = x.tiprel_1mes.fillna(-999).astype(np.int16)
    
    # indresi
    print('indresi')
    x.loc[x.indresi=='S','indresi'] = 1
    x.loc[x.indresi=='N','indresi'] = 0
    x['indresi'] = x.indresi.fillna(-999).astype(np.int16)
    
    # indext
    print('indext')
    x.loc[x.indext.isnull(),'indext'] = -999
    x.loc[x.indext=='N','indext'] = 0
    x.loc[x.indext=='S','indext'] = 1
    x['indext'] = x.indext.astype(np.int16)
    
    # conyuemp
    print('conyuemp')
    d_conyuemp = dict((k,v) for k,v in zip(x.conyuemp.value_counts().index,rankdata(x.conyuemp.value_counts(),method='ordinal')))
    x['conyuemp'] = x.conyuemp.map(d_conyuemp).fillna(-999).astype(np.int16)
    
    # canal_entrada
    print('canal_entrada')
    x['ce_startletter'] = x.canal_entrada.str[0]
    d_ce_startletter = dict((k,v) for k,v in zip(x.ce_startletter.value_counts().index,rankdata(x.ce_startletter.value_counts(),method='ordinal')))
    x['ce_startletter'] = x.ce_startletter.map(d_ce_startletter).fillna(-999).astype(np.int16)
    x['ce_middleletter'] =  x.canal_entrada.str[1]
    d_ce_middleletter = dict((k,v) for k,v in zip(x.ce_middleletter.value_counts().index,rankdata(x.ce_middleletter.value_counts(),method='ordinal')))
    x['ce_middleletter'] = x.ce_middleletter.map(d_ce_middleletter).fillna(-999).astype(np.int16)
    x['ce_endletter'] = x.canal_entrada.str[2]
    d_ce_endletter = dict((k,v) for k,v in zip(x.ce_endletter.value_counts().index,rankdata(x.ce_endletter.value_counts(),method='ordinal')))
    x['ce_endletter'] = x.ce_endletter.map(d_ce_endletter).fillna(-999).astype(np.int16)
    
    d_canal_entrada = dict((k,v) for k,v in zip(x.canal_entrada.value_counts().index,rankdata(x.canal_entrada.value_counts(),method='ordinal')))
    x['canal_entrada'] = x.canal_entrada.map(d_canal_entrada).fillna(-999).astype(np.int16)
    
    # indfall
    print('indfall')
    x.loc[x.indfall.isnull(),'indfall'] = -999 #27k
    x.loc[x.indfall=='N','indfall'] = 0
    x.loc[x.indfall=='S','indfall'] = 1
    x['indfall'] = x.indfall.fillna(-999).astype(np.int16)
    
    # tipodom
    print('tipodom')
    x.drop('tipodom',axis=1,inplace=True)
    
    # cod_prov, nomprov
    print('cod_prov,nomprov')
    x['cod_prov'] = pd.to_numeric(x.cod_prov,errors='coerce').fillna(-999).astype(np.int16)
    
    
    # ind_actividad_cliente
    print('ind_actividad_cliente')
    x.loc[x.ind_actividad_cliente.isnull(),'ind_actividad_cliente'] = -999 #27k, no idea
    x['ind_actividad_cliente'] = x.ind_actividad_cliente.astype(np.int16)
    
    # segmento
    print('segmento')
    d_segmento = dict((k,v) for k,v in zip(x.segmento.value_counts().index,rankdata(x.segmento.value_counts(),method='ordinal')))
    x['segmento'] = x.segmento.map(d_segmento).fillna(-999).astype(np.int16)
    
    # renta
    print('renta')
    x['renta'] = pd.to_numeric(x.renta,errors='coerce').fillna(-999)
    x['renta'] = x.renta.astype(np.float32)
    
    # Additional features
    print('Additional features')
    # Number of accounts per nomprov per fecha_dato
    # Median salary for nomprov per fecha_dato
    # Median age for nomprov per fecha_dato
    
    # Contract length
    print('contract length')
    x['contract_length'] = (pd.to_datetime(x.fecha_dato)-x.fecha_alta).dt.days.astype(np.int16)
    x.loc[x.contract_length<0,'contract_length'] = 0
    
    # Year born
    print('year born')
    x['year_born'] = (pd.to_datetime(x.fecha_dato) - pd.to_timedelta(x.age.astype(np.int32)*365, unit='D')).dt.year.astype(np.int16)
    
    # Other that were previously in the csv dict version
    print('contract percentage, month')
    x['contract_percentage_age'] = (x.contract_length / (x.age.astype(np.int32)*365)).fillna(-999).astype(np.float32)
    x['contract_percentage_antiguedad'] = ( (x.contract_length - (x.antiguedad.astype(np.int32)*12)) / (x.contract_length) ).fillna(-999).astype(np.float32)
    x.loc[x.contract_percentage_antiguedad.isnull(),'contract_percentage_antiguedad'] = -999
    x['month'] = pd.to_datetime(x.fecha_dato).dt.month
    
    # Products
    print('products')
    # Missing products (maybe didn't exist back then, all <= 2015-06-28
    x.loc[(x.fecha_dato<'2016-06-28') & (x.ind_nomina_ult1.isnull()) & (x.ind_cno_fin_ult1==1),['ind_nomina_ult1','ind_nom_pens_ult1']] = -999
    x.loc[(x.fecha_dato<'2016-06-28') & (x.ind_nomina_ult1.isnull()) & (x.ind_cno_fin_ult1==0),['ind_nomina_ult1','ind_nom_pens_ult1']] = -999
    
    # Test set products
    x.loc[x[prod_features()].isnull().any(1),prod_features()] = -999
    x[prod_features()] = x[prod_features()].fillna(-999).astype(np.int16)
    
    # Previous products to current line
    print('products')
    x.sort_values(['ncodpers','fecha_dato'],inplace=True)
    msk = x.ncodpers != x.ncodpers.shift(1)
    #original = x.loc[msk,prod_features()]
    x[prod_features()] = x[prod_features()].shift(1)
    x.loc[msk,prod_features()] = np.nan#original
    x[prod_features()] = x[prod_features()].fillna(-999).astype(np.int16)
    
    # Add cumsum?
    
    # Products added since last
    print('products added or removed')
    change_cols = [name+"_change" for name in prod_features()]
    x[change_cols] = x[prod_features()] - x[prod_features()].shift(1)
    x.loc[msk,change_cols] = np.nan
    x.loc[(x[change_cols]>1).any(1),change_cols] = -999
    x[change_cols] = x[change_cols].fillna(-999).astype(np.int16)
    
    # Number of products added
    x['num_products_added'] = (x[change_cols]==1).sum(1)
    x['num_products_removed'] = (x[change_cols]==-1).sum(1)
    
    # Number of products in "product groups"
    print('product groups')
    x['num_rare'] = x['ind_ahor_fin_ult1'] + x['ind_aval_fin_ult1']
    x['num_particular'] = x['ind_ctma_fin_ult1'] + x['ind_ctop_fin_ult1'] + x['ind_ctpp_fin_ult1']
    x['num_deposits'] = x['ind_deco_fin_ult1'] + x['ind_deme_fin_ult1'] + x['ind_dela_fin_ult1']
    x['num_pension'] =  x['ind_nomina_ult1'] + x['ind_nom_pens_ult1']
    x['num_finance'] = x['ind_fond_fin_ult1'] + x['ind_valo_fin_ult1'] + x['ind_cder_fin_ult1'] + x['ind_plan_fin_ult1']
    x['num_credit'] = x['ind_recibo_ult1'] + x['ind_tjcr_fin_ult1']
    x['num_home'] =  x['ind_viv_fin_ult1'] + x['ind_hip_fin_ult1'] + x['ind_pres_fin_ult1']
    x['num_work'] = x['ind_cno_fin_ult1'] + x['ind_reca_fin_ult1']
    x['num_modern'] = x['ind_ecue_fin_ult1'] + x['ind_ctju_fin_ult1']
    cols = ['num_rare','num_particular','num_deposits','num_pension','num_finance','num_credit','num_home','num_work','num_modern']
    x.loc[(x[cols]<0).any(1),cols] = -999
    x[cols] = x[cols].fillna(-999).astype(np.int16)
    
    # Indicators for change
    print('change indicators')
    msk = x.ncodpers != x.ncodpers.shift(1)
    
    x['ind_actividad_cliente_diff'] = x.ind_actividad_cliente - x.ind_actividad_cliente.shift(1)
    x.loc[msk,'ind_actividad_cliente_diff'] = 0
    x['ind_actividad_cliente_diff'] = x.ind_actividad_cliente_diff.fillna(-999).astype(np.int16)
    
    x['ind_nuevo_diff'] = x.ind_nuevo - x.ind_nuevo.shift(1)
    x.loc[msk,'ind_nuevo_diff'] = 0
    x['ind_nuevo_diff'] = x.ind_nuevo_diff.fillna(-999).astype(np.int16)
    
    x['cod_prov_diff'] = x.cod_prov - x.cod_prov.shift(1)
    x.loc[msk,'cod_prov_diff'] = 0
    x['cod_prov_diff'] = x.cod_prov_diff.fillna(-999).astype(np.int16)
    
    x['segmento_diff'] = x.segmento - x.segmento.shift(1)
    x.loc[msk,'segmento_diff'] = 0
    x['segmento_diff'] = x.segmento_diff.fillna(-999).astype(np.int16)
    
    x['indrel_diff'] = x.indrel - x.indrel.shift(1)
    x.loc[msk,'indrel_diff'] = 0
    x['indrel_diff'] = x.indrel_diff.fillna(-999).astype(np.int16)
    
    x['indrel_1mes_diff'] = x.indrel_1mes - x.indrel_1mes.shift(1)
    x.loc[msk,'indrel_1mes_diff'] = 0
    x['indrel_1mes_diff'] = x.indrel_1mes_diff.fillna(-999).astype(np.int16)
    
    x['tiprel_1mes_diff'] = x.tiprel_1mes - x.tiprel_1mes.shift(1)
    x.loc[msk,'tiprel_1mes_diff'] = 0
    x['tiprel_1mes_diff'] = x.tiprel_1mes_diff.fillna(-999).astype(np.int16)
    
    x['canal_entrada_diff'] = x.canal_entrada - x.canal_entrada.shift(1)
    x.loc[msk,'canal_entrada_diff'] = 0
    x['canal_entrada_diff'] = x.canal_entrada_diff.fillna(-999).astype(np.int16)
    
    x['age_diff'] = x.age - x.age.shift(1)
    x.loc[msk,'age_diff'] = 0
    x['age_diff'] = x.age_diff.fillna(-999).astype(np.int16)
    
    # Belongs to two of the most common indicators
    for col in ['cod_prov','segmento','indrel_1mes','tiprel_1mes','canal_entrada']:
        name = col+'_most_common'
        x[name] = (x[col] >= (x[col].max())).fillna(-999).astype(np.int8)
    
    endshape = x.shape[0]
    print('End shape',endshape)
    
    assert startshape==endshape
    
    return(x)
    
# Preprocess function 3 
def preprocess3(x):
    
    startshape = x.shape[0]
    
    print('Start shape',startshape)
    
    # Age
    x['age'] = pd.to_numeric(x.age,errors='coerce')
    
    # Null ages that exist
    nc = x.loc[x.age.isnull(),'ncodpers'].unique()
    nc2 = x[x.ncodpers.isin(nc) & x.age.notnull()].ncodpers.unique()
    age = x.loc[x.ncodpers.isin(nc2),['ncodpers','fecha_dato','age']]
    age['dummy'] = 100
    age.loc[age.age.notnull(),'dummy'] = age.age
    age = age.sort_values(['ncodpers','fecha_dato'],ascending=[1,0]).groupby('ncodpers',as_index=False).cummin().sort_values(['ncodpers','fecha_dato'])
    x.loc[age.index,'age'] = age.dummy
    
    # Rest are null
    x['age'] = x.age.fillna(-999).astype(np.int16)
    
    # Has the age changed? This could tell about customer activity (been in contact with bank and informed about current age)
    nunique_ages = pd.DataFrame(x.groupby('ncodpers')['age'].nunique()).astype(np.int8)
    nunique_ages.columns = ['nunique_ages']
    x = x.merge(nunique_ages,left_on='ncodpers',right_index=True,how='left')
    
    
    # Antiguedad, make running from max
    x['antiguedad'] = pd.to_numeric(x.antiguedad,errors='coerce').fillna(-999).astype(np.int16)
    ant = x[['ncodpers','fecha_dato','antiguedad']].copy()
    ant['real'] = 1
    ant['real2'] = ant.groupby('ncodpers')['real'].cumsum()-1
    tmp = ant['real2'].values.tolist()
    ant.sort_values(['ncodpers','fecha_dato'],ascending=[1,0],inplace=True)
    ant['tmp'] = tmp
    antmax = pd.DataFrame(ant.groupby('ncodpers')['antiguedad'].max())
    antmax.columns = ['antmax']
    ant = ant.merge(antmax,left_on='ncodpers',right_index=True,how='left')
    ant['antiguedad_real'] = ant.antmax-ant.tmp
    ant = ant.sort_values(['ncodpers','fecha_dato'])[['antiguedad_real']]
    x['antiguedad'] = ant['antiguedad_real']
    x.loc[x.antiguedad<-100,'antiguedad'] = -999
    
    # Make everything antiguedad to start from 0
    nc = x.loc[(x.antiguedad<0) & (x.antiguedad>-99)].ncodpers.unique()
    bad = x.loc[x.ncodpers.isin(nc),['ncodpers','antiguedad']]
    bad = -bad.groupby('ncodpers').min()
    bad = x.loc[x.ncodpers.isin(nc),['ncodpers','antiguedad']].merge(bad,left_on='ncodpers',right_index=True,how='left')
    bad['antiguedad'] = bad['antiguedad_x'] + bad['antiguedad_y']
    x.loc[x.ncodpers.isin(nc),'antiguedad'] = bad['antiguedad']
    x['antiguedad'] = x.antiguedad.astype(np.int16)
    
    
    # Renta
    x['renta'] = pd.to_numeric(x.renta,errors='coerce').astype(np.float32)
    
    # Fill those that have existing renta
    nc = x[x.renta.isnull()].ncodpers.unique()
    good = x.loc[(x.ncodpers.isin(nc)) & (x.renta.notnull())].ncodpers.unique()
    x.loc[x.ncodpers.isin(good)].groupby('ncodpers')['renta'].first()
    renta = x.loc[x.ncodpers.isin(good)].groupby('ncodpers')[['renta']].first()
    renta = x.loc[x.ncodpers.isin(good),['ncodpers']].merge(renta,left_on='ncodpers',right_index=True,how='left')
    x.loc[x.ncodpers.isin(good),'renta'] = renta.renta
    
    # Rest is unknown
    x['renta'] = x['renta'].fillna(-999).astype(np.float32)
    
    
    # tipodom
    x.drop('tipodom',axis=1,inplace=True)
    
    
    # Categorical encoding
    x.loc[x.indrel_1mes=='P','indrel_1mes'] = 5
    x['indrel_1mes'] = pd.to_numeric(x.indrel_1mes,errors='coerce')
    for col in ['canal_entrada','cod_prov','conyuemp','ind_actividad_cliente','ind_empleado',
                'indext','indfall','indrel','indrel_1mes','indresi',
                'nomprov','pais_residencia','segmento','sexo','tiprel_1mes',
                'ind_nuevo']:
        print(col)
        valc = x[col].value_counts()
        d = dict((k,v) for k,v in zip(valc.index,rankdata(valc,method='ordinal')))
        x[col] = x[col].map(d).fillna(-999).astype(np.int16)
    
    
    # fecha_alta
    print('fecha_alta')
    x['fecha_alta'] = pd.to_datetime(x.fecha_alta,errors='coerce').astype(np.int64) # 27k
    x.loc[x.fecha_alta < 0,'fecha_alta'] = np.nan
    fecha_alta = x.groupby('ncodpers')[['fecha_alta']].first()
    fecha_alta = fecha_alta.interpolate()
    fecha_alta = x.loc[x.fecha_alta.isnull(),['ncodpers','fecha_alta']].merge(fecha_alta,left_on='ncodpers',right_index=True,how='inner')
    fecha_alta = fecha_alta.reindex(x.loc[x.fecha_alta.isnull()].index)
    x.loc[x.fecha_alta.isnull(),'fecha_alta'] = fecha_alta['fecha_alta_y']
    x['fecha_alta'] = pd.to_datetime(x.fecha_alta,errors='coerce')
    
    
    # fecha_dato
    x['fecha_dato'] = pd.to_datetime(x.fecha_dato,format='%Y-%m-%d',errors='coerce')
    
    
    # ult_fec_cli_1t
    print('ult_fec_cli_1t')
    x['ult_fec_cli_1t'] = pd.to_datetime(x.ult_fec_cli_1t,format='%Y-%m-%d',errors='coerce')
    x['primary_days_left'] = (x.ult_fec_cli_1t - x.fecha_dato).dt.days + 31
    x.loc[x.primary_days_left.isnull(),'primary_days_left'] = 0
    x['primary_days_left'] = x.primary_days_left.astype(np.int16)
    x['ult_fec_cli_1t'] = x.ult_fec_cli_1t.astype(str)
    x.loc[x.ult_fec_cli_1t=='NaT','ult_fec_cli_1t'] = '0'
    x.loc[x.ult_fec_cli_1t!='NaT','ult_fec_cli_1t'] = '1'
    x['ult_fec_cli_1t'] = pd.to_numeric(x.ult_fec_cli_1t,errors='coerce')
    x['ult_fec_cli_1t'] = x.ult_fec_cli_1t.fillna(-999).astype(np.int16)
    
    
    # Products
    print('products')
    # Missing products (maybe didn't exist back then, all <= 2015-06-28
    x.loc[(x.fecha_dato<'2016-06-28') & (x.ind_nomina_ult1.isnull()) & (x.ind_cno_fin_ult1==1),['ind_nomina_ult1','ind_nom_pens_ult1']] = -999
    x.loc[(x.fecha_dato<'2016-06-28') & (x.ind_nomina_ult1.isnull()) & (x.ind_cno_fin_ult1==0),['ind_nomina_ult1','ind_nom_pens_ult1']] = -999
    
    # Test set products
    x.loc[x[prod_features()].isnull().any(1),prod_features()] = -999
    x[prod_features()] = x[prod_features()].fillna(-999).astype(np.int8)
    
    # Previous products to current line
    x.sort_values(['ncodpers','fecha_dato'],inplace=True)
    msk = x.ncodpers != x.ncodpers.shift(1)
    x[prod_features()] = x[prod_features()].shift(1)
    x.loc[msk,prod_features()] = np.nan
    x[prod_features()] = x[prod_features()].fillna(-999).astype(np.int8)
    
    # Products added since last
    print('products added or removed')
    change_cols = [name+"_change" for name in prod_features()]
    x[change_cols] = x[prod_features()] - x[prod_features()].shift(1)
    x.loc[msk,change_cols] = np.nan
    x.loc[(x[change_cols]>1).any(1),change_cols] = -999
    x[change_cols] = x[change_cols].fillna(-999).astype(np.int8)
    
    # Number of products added
    x['num_products_added'] = ((x[change_cols]==1).sum(1)).astype(np.int16)
    x['num_products_removed'] = ((x[change_cols]==-1).sum(1)).astype(np.int16)
    
    # Number of products in "product groups"
    print('product groups')
    x['num_rare'] = x['ind_ahor_fin_ult1'] + x['ind_aval_fin_ult1']
    x['num_particular'] = x['ind_ctma_fin_ult1'] + x['ind_ctop_fin_ult1'] + x['ind_ctpp_fin_ult1']
    x['num_deposits'] = x['ind_deco_fin_ult1'] + x['ind_deme_fin_ult1'] + x['ind_dela_fin_ult1']
    x['num_pension'] =  x['ind_nomina_ult1'] + x['ind_nom_pens_ult1']
    x['num_finance'] = x['ind_fond_fin_ult1'] + x['ind_valo_fin_ult1'] + x['ind_cder_fin_ult1'] + x['ind_plan_fin_ult1']
    x['num_credit'] = x['ind_recibo_ult1'] + x['ind_tjcr_fin_ult1']
    x['num_home'] =  x['ind_viv_fin_ult1'] + x['ind_hip_fin_ult1'] + x['ind_pres_fin_ult1']
    x['num_work'] = x['ind_cno_fin_ult1'] + x['ind_reca_fin_ult1']
    x['num_modern'] = x['ind_ecue_fin_ult1'] + x['ind_ctju_fin_ult1']
    cols = ['num_rare','num_particular','num_deposits','num_pension','num_finance','num_credit','num_home','num_work','num_modern']
    x.loc[(x[cols]<0).any(1),cols] = -999
    x[cols] = x[cols].fillna(-999).astype(np.int8)
    
    
    # Additional features
    print('additional features')
    # Differences
    msk = x.ncodpers != x.ncodpers.shift(1)
    for col in ['canal_entrada','cod_prov','conyuemp','ind_actividad_cliente','ind_empleado',
                'indext','indfall','indrel','indrel_1mes','indresi',
                'nomprov','pais_residencia','segmento','sexo','tiprel_1mes',
                'ind_nuevo']:
        name = col+"_diff"
        print(name)
        x[name] = x[col] - x[col].shift(1)
        x.loc[msk,name] = 0
        x[name] = x[name].fillna(-999).astype(np.int16)
        
    # Contract length
    x['contract_length'] = (x.fecha_dato - x.fecha_alta).dt.days
    x.loc[x.contract_length<-10,'contract_length'] = np.nan
    x.loc[x.contract_length<0,'contract_length'] = 0
    x['contract_percentage_age'] = (x.contract_length / (x.age.astype(np.int32)*365))
    x.loc[x.contract_percentage_age>1,'contract_percentage_age'] = 1
    x.loc[x.contract_percentage_age<0,'contract_percentage_age'] = np.nan
    x['contract_percentage_antiguedad'] = ( (x.contract_length - (x.antiguedad.astype(np.int32)*30)) / (x.contract_length) )
    
    x['contract_length'] = x.contract_length.fillna(-999).astype(np.int16)
    x['contract_percentage_age'] = x.contract_percentage_age.fillna(-999).replace(np.Inf,-999).replace(-np.Inf,-999).astype(np.float32)
    x['contract_percentage_antiguedad'] = x.contract_percentage_antiguedad.fillna(-999).replace(np.Inf,-999).replace(-np.Inf,-999).astype(np.float32)
    
    # Date features
    x['fecha_dato_month'] = x.fecha_dato.dt.month
    x['fecha_alta_month'] = x.fecha_alta.dt.month
    x['fecha_alta_year'] = x.fecha_alta.dt.year
    
    # Cumulative added/removed
    cm = x.groupby('ncodpers')[['num_products_added','num_products_removed']].cumsum()
    cm.columns = ['num_products_added_cumsum','num_products_removed_cumsum']
    x['num_products_added_cumsum'] = cm['num_products_added_cumsum']
    x['num_products_removed_cumsum'] = cm['num_products_removed_cumsum']
    
    # Median renta
    median_renta = x[x.renta!=-999].groupby(['cod_prov','fecha_dato','segmento'])['renta'].median().reset_index()
    median_renta.columns = ['cod_prov','fecha_dato','segmento','median_renta']
    x['median_renta'] = x[['cod_prov','fecha_dato','segmento']].merge(median_renta,on=['cod_prov','fecha_dato','segmento'],how='left')['median_renta']
    median_renta = x[x.renta!=-999].groupby(['fecha_dato'])['renta'].median().reset_index()
    median_renta.columns = ['fecha_dato','median_renta']
    median_renta = x[['fecha_dato']].merge(median_renta,on=['fecha_dato'],how='left')
    x.loc[x.median_renta.isnull(),'median_renta'] = median_renta[x.median_renta.isnull()]
    x['ratio_renta'] = (x.renta / x.median_renta)
    
    endshape = x.shape[0]

    print('End shape',endshape)
    
    assert startshape == endshape

    
    return(x)
    
# Preprocess function for LSTM, with one-hot-encoded features
def preprocess_lstm(x):
    
    startshape = x.shape[0]
    print(startshape)
    
    def get_age():
        # Age
        x['age'] = pd.to_numeric(x.age,errors='coerce')
        
        # Null ages that exist
        nc = x.loc[x.age.isnull(),'ncodpers'].unique()
        nc2 = x[x.ncodpers.isin(nc) & x.age.notnull()].ncodpers.unique()
        age = x.loc[x.ncodpers.isin(nc2),['ncodpers','fecha_dato','age']]
        age['dummy'] = 100
        age.loc[age.age.notnull(),'dummy'] = age.age
        age = age.sort_values(['ncodpers','fecha_dato'],ascending=[1,0]).groupby('ncodpers',as_index=False).cummin().sort_values(['ncodpers','fecha_dato'])
        x.loc[age.index,'age'] = age.dummy
        
        # Rest are null
        x['age'] = x.age.fillna(x.age.median()).astype(np.int16)
    
    def coerce_numeric(x,col):
        print('coerce numeric',col)
        x[col] = pd.to_numeric(x[col],errors='coerce').astype(np.float32)
        x.loc[x[col]<0,col] = 0
        return(x)
    
    def coerce_dates(x,col):
        print('coerce dates',col)
        x[col] = pd.to_datetime(x[col],errors='coerce')
        return(x)
    
    def assign_cat_to_most_common(x,col):
        print('assign nan',col)
        most_common = x[col].value_counts().index[0]
        x.loc[x[col].isnull(),col] = most_common
        return(x)
    
    def assign_median(x,col):
        print('assign median',col)
        x.loc[x[col].isnull(),col] = x[col].median()
        return(x)
    
    def dmap(x,col,dtype=np.int16):
        print('dmap',col)
        valc = x[col].value_counts()
        d = dict((k,v) for k,v in zip(valc.index,rankdata(valc,method='ordinal')))
        x[col] = x[col].map(d).astype(dtype)
        return(x)
    
    def top_cats(x,col,top=5):
        print('top cats',col)
        mc = x[col].value_counts()
        n_mc = mc.shape[0]
        x.loc[x[col] < (n_mc-top),col] = n_mc-top-1
        return(x)
    
    def oh_encode(x,col):
        print('oh encode',col)
        vec = x[col].values
        lb = LabelBinarizer()
        lb.fit(vec)
        vec = lb.transform(vec)
        names = [col+"_"+str(i) for i in range(vec.shape[1])]
        df = pd.DataFrame(vec,columns=names,dtype=np.int8)
        x = pd.concat([x,df],axis=1)
        x = x.drop(col,axis=1)
        return(x)
    
    def handle_products(x):
        print('products')
        # Missing products (maybe didn't exist back then, all <= 2015-06-28
        x.loc[(x.fecha_dato<'2016-06-28') & (x.ind_nomina_ult1.isnull()) & (x.ind_cno_fin_ult1==1),['ind_nomina_ult1','ind_nom_pens_ult1']] = 1.0
        x.loc[(x.fecha_dato<'2016-06-28') & (x.ind_nomina_ult1.isnull()) & (x.ind_cno_fin_ult1==0),['ind_nomina_ult1','ind_nom_pens_ult1']] = 0.0
        
        # Test set products
        x.loc[x[prod_features()].isnull().any(1),prod_features()] = 0
        x[prod_features()] = x[prod_features()].astype(np.int8)
        
        # Previous products to current line
        x.sort_values(['ncodpers','fecha_dato'],inplace=True)
        msk = x.ncodpers != x.ncodpers.shift(1)
        x[prod_features()] = x[prod_features()].shift(1)
        x.loc[msk,prod_features()] = np.nan
        x[prod_features()] = x[prod_features()].fillna(-1).astype(np.int8)
        return(x)
        
    def add_features(x):
        print('adding features')
        x['month'] = x.fecha_dato.dt.month.astype(np.int8)
        
        x['contract_length'] = (x.fecha_dato - x.fecha_alta).dt.days
        x.loc[x.contract_length.isnull(),'contract_length'] = x.contract_length.median()
        x['contract_length'] = x.contract_length.astype(np.float32)
        return(x)
        
        
    x.drop('tipodom',inplace=True,axis=1)
        
    print(x.dtypes)
    
    # Numerical
    for col in ['age','antiguedad','renta']:
        x = coerce_numeric(x,col)
        x = assign_median(x,col)
    print(x.dtypes)
    
    # Dates
    for col in ['fecha_dato','fecha_alta','ult_fec_cli_1t']:
        x = coerce_dates(x,col) 
    print(x.dtypes)
    
    # Categorical
    for col in ['cod_prov','ind_nuevo','ind_actividad_cliente','ind_empleado','indext','indfall', 'canal_entrada','conyuemp',
                'indrel','indrel_1mes','indresi','nomprov','pais_residencia','segmento','sexo','tiprel_1mes']:
        x = assign_cat_to_most_common(x, col)
        x = dmap(x,col)
        x = top_cats(x,col)
        x = oh_encode(x,col)
        gc.collect()
    
    print(x.dtypes)
    
    # Products
    x = handle_products(x)
    
    print(x.dtypes)
    
    # Additional features
    x = add_features(x)
    
    print(x.dtypes)
    
    endshape = x.shape[0]
    
    
    print('End shape',endshape)
    
    assert startshape==endshape
    
    return(x)

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# All of the preprocessing functions can be processed further processed like this, if not enough RAM
# Below is just an example (create_features, process_user)
def create_features():
    create_folder(bp()+'/data/features/')
    with open(bp()+'/data/preprocessed/x.csv','r') as f:
        cols = f.readline().split(",")
        nc_ind = cols.index('ncodpers')
        first_x = f.readline().split(",")
        first_nc = first_x[nc_ind]
    
    mp = 0
    counter = 0
    n_processed = 0
    n_ncodpers = 0
    current_nc = first_nc
    x = []
    infile = open(bp()+'/data/preprocessed/x.csv')
    outfile = open(bp()+'/data/features/x.csv','w')
    outfile.write(",".join(features()) + "\n")
    for r in csv.DictReader(infile):
        
        new_nc = r['ncodpers']

        # Process all material for one user
        if new_nc != current_nc:
            process_user(x,outfile)
            x = []
            current_nc = new_nc
            n_ncodpers += 1
            
        x.append(r)
        
        
        if counter % 100000 == 0:
            print('Rows %d | users %d' % (mp*100000,n_ncodpers))
            mp += 1
            counter = 0
        n_processed += 1
        counter += 1
    
    # Last user
    process_user(x,outfile)
    infile.close()
    outfile.close()  
      
def process_user(x_in,outfile):
    
    pr = x_in[0]
    ppr = pr
    
    # INIT
    ncodpers = pr['ncodpers']
    
    min_renta = float(pr['renta'])
    max_renta = float(pr['renta'])
    cum_renta = 0
    
    fecha_alta = pd.to_datetime([e['fecha_alta'] for e in x_in],format='%Y-%m-%d',errors='coerce').min()
    
    
    # ITERATE
    for t,r in enumerate(x_in):
        
        # Fecha dato
        fecha_dato = r['fecha_dato']
        fecha_dato_dt = pd.to_datetime(fecha_dato,format='%Y-%m-%d')
        
        # Basic numeric
        age = int(r['age'])
        renta = float(r['renta'])
        antiguedad = int(r['antiguedad'])
        
        # Renta specific
        min_renta = min(renta,min_renta)
        max_renta = max(renta,max_renta)
        cum_renta += renta
        diff_renta = renta - float(pr['renta'])
        diff_renta_percentage = min([diff_renta / renta,30000000])
        ind_renta_diff_negative = int(diff_renta < 0)
        ratio_renta = renta / float(r['median_renta'])
        
        # Additional numeric
        ratio_age = age / float(r['median_age'])
        nunique_ncodpers = int(r['nunique_ncodpers'])
        contract_length = (fecha_dato_dt - fecha_alta).days
        if contract_length == 0:
            contract_length = -1
        contract_percentage_age = age*365/contract_length
        contract_percentage_antiguedad = (contract_length-antiguedad*12)/contract_length
        year_born = (fecha_dato_dt - pd.DateOffset(years=age)).year
        month = fecha_dato_dt.month
        n_nan = int(r['n_nan'])
        
        # Categorical variables
        ind_empleado = int(r['ind_empleado'])
        pais_residencia = int(r['pais_residencia'])
        sexo = int(r['sexo'])
        ind_nuevo = int(r['ind_nuevo'])
        indrel = int(r['indrel'])
        indrel_1mes = int(r['indrel_1mes'])
        tiprel_1mes = int(r['tiprel_1mes'])
        indresi = int(r['indresi'])
        indext = int(r['indext'])
        conyuemp = int(r['conyuemp'])
        canal_entrada = int(r['canal_entrada'])
        indfall = int(r['indfall'])
        cod_prov = int(r['cod_prov'])
        ind_actividad_cliente = int(r['ind_actividad_cliente'])
        segmento = int(r['segmento'])
        
        # Categorical different than most popular
        ind_empleado_most_common = int(ind_empleado==0)
        pais_residencia_most_common = int(pais_residencia==0)
        sexo_most_common = int(sexo==1)
        ind_nuevo_most_common = int(ind_nuevo==0)
        indrel_most_common = int(indrel==1)
        indrel_1mes_most_common = int(indrel_1mes==1)
        tiprel_1mes_most_common = int(tiprel_1mes==1)
        indresi_most_common = int(indresi==1)
        indext_most_common = int(indext==0)
        conyuemp_most_common = int(conyuemp==0)
        canal_entrada_most_common = int(canal_entrada==1)
        indfall_most_common = int(indfall==0)
        cod_prov_most_common = int(cod_prov==28)
        ind_actividad_cliente_most_common = int(ind_actividad_cliente==0)
        segmento_most_common = int(segmento==0)
        
        # Categorical difference from previous month
        ind_empleado_diff = int(ind_empleado!=pr['ind_empleado'])
        pais_residencia_diff = int(pais_residencia!=pr['pais_residencia'])
        sexo_diff = int(sexo!=pr['sexo'])
        ind_nuevo_diff = int(ind_nuevo!=pr['ind_nuevo'])
        indrel_diff = int(indrel!=pr['indrel'])
        indrel_1mes_diff = int(indrel_1mes!=pr['indrel_1mes'])
        tiprel_1mes_diff = int(tiprel_1mes!=pr['tiprel_1mes'])
        indresi_diff = int(indresi!=pr['indresi'])
        indext_diff = int(indext!=pr['indext'])
        conyuemp_diff = int(conyuemp!=pr['conyuemp'])
        canal_entrada_diff = int(canal_entrada!=pr['canal_entrada'])
        indfall_diff = int(indfall!=pr['indfall'])
        cod_prov_diff = int(cod_prov!=pr['cod_prov'])
        ind_actividad_cliente_diff = int(ind_actividad_cliente!=pr['ind_actividad_cliente'])
        segmento_diff = int(segmento!=pr['segmento'])
        
        # Products
        ind_ahor_fin_ult1 = int(pr['ind_ahor_fin_ult1'])    # Saving Account
        ind_aval_fin_ult1 = int(pr['ind_aval_fin_ult1'])    # Guarantees
        ind_cco_fin_ult1 = int(pr['ind_cco_fin_ult1'])     #Current Accounts
        ind_cder_fin_ult1 = int(pr['ind_cder_fin_ult1'])     #Derivada Account
        ind_cno_fin_ult1 = int(pr['ind_cno_fin_ult1'])     #Payroll Account
        ind_ctju_fin_ult1 = int(pr['ind_ctju_fin_ult1'])     #Junior Account
        ind_ctma_fin_ult1 = int(pr['ind_ctma_fin_ult1'])  #Mas particular Account
        ind_ctop_fin_ult1 = int(pr['ind_ctop_fin_ult1'])   #particular Account
        ind_ctpp_fin_ult1 = int(pr['ind_ctpp_fin_ult1'])  #particular Plus Account
        ind_deco_fin_ult1 = int(pr['ind_deco_fin_ult1'])  #Short-term deposits
        ind_deme_fin_ult1 = int(pr['ind_deme_fin_ult1'])   #Medium-term deposits
        ind_dela_fin_ult1 = int(pr['ind_dela_fin_ult1'])     #Long-term deposits
        ind_ecue_fin_ult1 = int(pr['ind_ecue_fin_ult1'])     #e-account
        ind_fond_fin_ult1 = int(pr['ind_fond_fin_ult1'])    #Funds
        ind_hip_fin_ult1 = int(pr['ind_hip_fin_ult1'])     #Mortgage
        ind_plan_fin_ult1 = int(pr['ind_plan_fin_ult1'])    #Pensions
        ind_pres_fin_ult1 = int(pr['ind_pres_fin_ult1'])     #Loans
        ind_reca_fin_ult1 = int(pr['ind_reca_fin_ult1'])    #Taxes
        ind_tjcr_fin_ult1 = int(pr['ind_tjcr_fin_ult1'])     #Credit Card
        ind_valo_fin_ult1 = int(pr['ind_valo_fin_ult1'])     #Securities
        ind_viv_fin_ult1 = int(pr['ind_viv_fin_ult1'])    #Home Account
        ind_nomina_ult1 = int(pr['ind_nomina_ult1'])     #Payroll
        ind_nom_pens_ult1 = int(pr['ind_nom_pens_ult1'])     #Pensions
        ind_recibo_ult1 = int(pr['ind_recibo_ult1'])     #Direct Debit
        
        # Products added / removed
        p_ind_ahor_fin_ult1 = int(ppr['ind_ahor_fin_ult1'])
        p_ind_aval_fin_ult1 = int(ppr['ind_aval_fin_ult1']) 
        p_ind_cco_fin_ult1 = int(ppr['ind_cco_fin_ult1'])     #Current Accounts
        p_ind_cder_fin_ult1 = int(ppr['ind_cder_fin_ult1'])     #Derivada Account
        p_ind_cno_fin_ult1 = int(ppr['ind_cno_fin_ult1'])     #Payroll Account
        p_ind_ctju_fin_ult1 = int(ppr['ind_ctju_fin_ult1'])     #Junior Account
        p_ind_ctma_fin_ult1 = int(ppr['ind_ctma_fin_ult1'])  #Mas particular Account
        p_ind_ctop_fin_ult1 = int(ppr['ind_ctop_fin_ult1'])   #particular Account
        p_ind_ctpp_fin_ult1 = int(ppr['ind_ctpp_fin_ult1'])  #particular Plus Account
        p_ind_deco_fin_ult1 = int(ppr['ind_deco_fin_ult1'])  #Short-term deposits
        p_ind_deme_fin_ult1 = int(ppr['ind_deme_fin_ult1'])   #Medium-term deposits
        p_ind_dela_fin_ult1 = int(ppr['ind_dela_fin_ult1'])     #Long-term deposits
        p_ind_ecue_fin_ult1 = int(ppr['ind_ecue_fin_ult1'])     #e-account
        p_ind_fond_fin_ult1 = int(ppr['ind_fond_fin_ult1'])    #Funds
        p_ind_hip_fin_ult1 = int(ppr['ind_hip_fin_ult1'])     #Mortgage
        p_ind_plan_fin_ult1 = int(ppr['ind_plan_fin_ult1'])    #Pensions
        p_ind_pres_fin_ult1 = int(ppr['ind_pres_fin_ult1'])     #Loans
        p_ind_reca_fin_ult1 = int(ppr['ind_reca_fin_ult1'])    #Taxes
        p_ind_tjcr_fin_ult1 = int(ppr['ind_tjcr_fin_ult1'])     #Credit Card
        p_ind_valo_fin_ult1 = int(ppr['ind_valo_fin_ult1'])     #Securities
        p_ind_viv_fin_ult1 = int(ppr['ind_viv_fin_ult1'])    #Home Account
        p_ind_nomina_ult1 = int(ppr['ind_nomina_ult1'])     #Payroll
        p_ind_nom_pens_ult1 = int(ppr['ind_nom_pens_ult1'])     #Pensions
        p_ind_recibo_ult1 = int(ppr['ind_recibo_ult1'])     #Direct Debit
        
        # Added
        ind_ahor_fin_ult1_added = max(ind_ahor_fin_ult1 - p_ind_ahor_fin_ult1,0)
        ind_aval_fin_ult1_added = max(ind_aval_fin_ult1 - p_ind_aval_fin_ult1,0)
        ind_cco_fin_ult1_added = max(ind_cco_fin_ult1 - p_ind_cco_fin_ult1,0)
        ind_cder_fin_ult1_added = max(ind_cder_fin_ult1 - p_ind_cder_fin_ult1,0)
        ind_cno_fin_ult1_added = max(ind_cno_fin_ult1 - p_ind_cno_fin_ult1,0)
        ind_ctju_fin_ult1_added = max(ind_ctju_fin_ult1 - p_ind_ctju_fin_ult1,0)
        ind_ctma_fin_ult1_added = max(ind_ctma_fin_ult1 - p_ind_ctma_fin_ult1,0)
        ind_ctop_fin_ult1_added = max(ind_ctop_fin_ult1 - p_ind_ctop_fin_ult1,0)
        ind_ctpp_fin_ult1_added = max(ind_ctpp_fin_ult1 - p_ind_ctpp_fin_ult1,0)
        ind_deco_fin_ult1_added = max(ind_deco_fin_ult1 - p_ind_deco_fin_ult1,0)
        ind_deme_fin_ult1_added = max(ind_deme_fin_ult1 - p_ind_deme_fin_ult1,0)
        ind_dela_fin_ult1_added = max(ind_dela_fin_ult1 - p_ind_dela_fin_ult1,0)
        ind_ecue_fin_ult1_added = max(ind_ecue_fin_ult1 - p_ind_ecue_fin_ult1,0)
        ind_fond_fin_ult1_added = max(ind_fond_fin_ult1 - p_ind_fond_fin_ult1,0)
        ind_hip_fin_ult1_added = max(ind_hip_fin_ult1 - p_ind_hip_fin_ult1,0)
        ind_plan_fin_ult1_added = max(ind_plan_fin_ult1 - p_ind_plan_fin_ult1,0)
        ind_pres_fin_ult1_added = max(ind_pres_fin_ult1 - p_ind_pres_fin_ult1,0) 
        ind_reca_fin_ult1_added = max(ind_reca_fin_ult1 - p_ind_reca_fin_ult1,0) 
        ind_tjcr_fin_ult1_added = max(ind_tjcr_fin_ult1 - p_ind_tjcr_fin_ult1,0) 
        ind_valo_fin_ult1_added = max(ind_valo_fin_ult1 - p_ind_valo_fin_ult1,0) 
        ind_viv_fin_ult1_added = max(ind_viv_fin_ult1 - p_ind_viv_fin_ult1,0) 
        ind_nomina_ult1_added = max(ind_nomina_ult1 - p_ind_nomina_ult1,0) 
        ind_nom_pens_ult1_added = max(ind_nom_pens_ult1 - p_ind_nom_pens_ult1,0) 
        ind_recibo_ult1_added = max(ind_recibo_ult1 - p_ind_recibo_ult1,0)
        num_products_added = ind_ahor_fin_ult1_added + ind_aval_fin_ult1_added + ind_cco_fin_ult1_added + ind_cder_fin_ult1_added + ind_cno_fin_ult1_added + ind_ctju_fin_ult1_added \
                            + ind_ctma_fin_ult1_added + ind_ctop_fin_ult1_added + ind_ctpp_fin_ult1_added + ind_deco_fin_ult1_added + ind_deme_fin_ult1_added \
                            + ind_dela_fin_ult1_added + ind_ecue_fin_ult1_added + ind_fond_fin_ult1_added + ind_hip_fin_ult1_added + ind_plan_fin_ult1_added \
                            + ind_pres_fin_ult1_added + ind_reca_fin_ult1_added + ind_tjcr_fin_ult1_added + ind_valo_fin_ult1_added + ind_viv_fin_ult1_added \
                            + ind_nomina_ult1_added + ind_nom_pens_ult1_added + ind_recibo_ult1_added
                            
        # Removed
        ind_ahor_fin_ult1_removed = -min(ind_ahor_fin_ult1 - p_ind_ahor_fin_ult1,0)
        ind_aval_fin_ult1_removed = -min(ind_aval_fin_ult1 - p_ind_aval_fin_ult1,0)
        ind_cco_fin_ult1_removed = -min(ind_cco_fin_ult1 - p_ind_cco_fin_ult1,0)
        ind_cder_fin_ult1_removed = -min(ind_cder_fin_ult1 - p_ind_cder_fin_ult1,0)
        ind_cno_fin_ult1_removed = -min(ind_cno_fin_ult1 - p_ind_cno_fin_ult1,0)
        ind_ctju_fin_ult1_removed = -min(ind_ctju_fin_ult1 - p_ind_ctju_fin_ult1,0)
        ind_ctma_fin_ult1_removed = -min(ind_ctma_fin_ult1 - p_ind_ctma_fin_ult1,0)
        ind_ctop_fin_ult1_removed = -min(ind_ctop_fin_ult1 - p_ind_ctop_fin_ult1,0)
        ind_ctpp_fin_ult1_removed = -min(ind_ctpp_fin_ult1 - p_ind_ctpp_fin_ult1,0)
        ind_deco_fin_ult1_removed = -min(ind_deco_fin_ult1 - p_ind_deco_fin_ult1,0)
        ind_deme_fin_ult1_removed = -min(ind_deme_fin_ult1 - p_ind_deme_fin_ult1,0)
        ind_dela_fin_ult1_removed = -min(ind_dela_fin_ult1 - p_ind_dela_fin_ult1,0)
        ind_ecue_fin_ult1_removed = -min(ind_ecue_fin_ult1 - p_ind_ecue_fin_ult1,0)
        ind_fond_fin_ult1_removed = -min(ind_fond_fin_ult1 - p_ind_fond_fin_ult1,0)
        ind_hip_fin_ult1_removed = -min(ind_hip_fin_ult1 - p_ind_hip_fin_ult1,0)
        ind_plan_fin_ult1_removed = -min(ind_plan_fin_ult1 - p_ind_plan_fin_ult1,0)
        ind_pres_fin_ult1_removed = -min(ind_pres_fin_ult1 - p_ind_pres_fin_ult1,0) 
        ind_reca_fin_ult1_removed = -min(ind_reca_fin_ult1 - p_ind_reca_fin_ult1,0) 
        ind_tjcr_fin_ult1_removed = -min(ind_tjcr_fin_ult1 - p_ind_tjcr_fin_ult1,0) 
        ind_valo_fin_ult1_removed = -min(ind_valo_fin_ult1 - p_ind_valo_fin_ult1,0) 
        ind_viv_fin_ult1_removed = -min(ind_viv_fin_ult1 - p_ind_viv_fin_ult1,0) 
        ind_nomina_ult1_removed = -min(ind_nomina_ult1 - p_ind_nomina_ult1,0) 
        ind_nom_pens_ult1_removed = -min(ind_nom_pens_ult1 - p_ind_nom_pens_ult1,0) 
        ind_recibo_ult1_removed = -min(ind_recibo_ult1 - p_ind_recibo_ult1,0)
        num_products_removed = ind_ahor_fin_ult1_removed + ind_aval_fin_ult1_removed + ind_cco_fin_ult1_removed + ind_cder_fin_ult1_removed + ind_cno_fin_ult1_removed + ind_ctju_fin_ult1_removed \
                            + ind_ctma_fin_ult1_removed + ind_ctop_fin_ult1_removed + ind_ctpp_fin_ult1_removed + ind_deco_fin_ult1_removed + ind_deme_fin_ult1_removed \
                            + ind_dela_fin_ult1_removed + ind_ecue_fin_ult1_removed + ind_fond_fin_ult1_removed + ind_hip_fin_ult1_removed + ind_plan_fin_ult1_removed \
                            + ind_pres_fin_ult1_removed + ind_reca_fin_ult1_removed + ind_tjcr_fin_ult1_removed + ind_valo_fin_ult1_removed + ind_viv_fin_ult1_removed \
                            + ind_nomina_ult1_removed + ind_nom_pens_ult1_removed + ind_recibo_ult1_removed
        
        # Number of products in categories
        num_rare = ind_ahor_fin_ult1 + ind_aval_fin_ult1
        num_particular = ind_ctma_fin_ult1 + ind_ctop_fin_ult1 + ind_ctpp_fin_ult1
        num_deposits = ind_deco_fin_ult1 + ind_deme_fin_ult1 + ind_dela_fin_ult1
        num_pension =  ind_nomina_ult1 + ind_nom_pens_ult1
        num_finance = ind_fond_fin_ult1 + ind_valo_fin_ult1 + ind_cder_fin_ult1 + ind_plan_fin_ult1
        num_credit = ind_recibo_ult1 + ind_tjcr_fin_ult1
        num_home =  ind_viv_fin_ult1 + ind_hip_fin_ult1 + ind_pres_fin_ult1
        num_work = ind_cno_fin_ult1 + ind_reca_fin_ult1
        num_modern = ind_ecue_fin_ult1 + ind_ctju_fin_ult1
        num_products = num_particular + num_deposits + num_finance + num_credit + num_home + num_work + num_pension + num_modern + ind_cco_fin_ult1
        
        
        
        v = [
            
            ncodpers,fecha_dato,
            
            age,renta,antiguedad,
            
            min_renta,max_renta,cum_renta,diff_renta,diff_renta_percentage,ind_renta_diff_negative,ratio_renta,
            
            ratio_age,nunique_ncodpers,contract_length,contract_percentage_age,contract_percentage_antiguedad,year_born,month,n_nan,
            
            ind_empleado,pais_residencia,sexo,ind_nuevo,indrel,indrel_1mes,tiprel_1mes,indresi,indext,
            conyuemp,canal_entrada,indfall,cod_prov,ind_actividad_cliente,segmento,
            
            ind_empleado_most_common,pais_residencia_most_common,sexo_most_common,ind_nuevo_most_common,indrel_most_common,indrel_1mes_most_common,tiprel_1mes_most_common,indresi_most_common,indext_most_common,
            conyuemp_most_common,canal_entrada_most_common,indfall_most_common,cod_prov_most_common,ind_actividad_cliente_most_common,segmento_most_common,
            
            ind_empleado_diff,pais_residencia_diff,sexo_diff,ind_nuevo_diff,indrel_diff,indrel_1mes_diff,tiprel_1mes_diff,indresi_diff,indext_diff,
            conyuemp_diff,canal_entrada_diff,indfall_diff,cod_prov_diff,ind_actividad_cliente_diff,segmento_diff,
            
            ind_ahor_fin_ult1,ind_aval_fin_ult1,ind_cco_fin_ult1,ind_cder_fin_ult1,ind_cno_fin_ult1,ind_ctju_fin_ult1,ind_ctma_fin_ult1,ind_ctop_fin_ult1,ind_ctpp_fin_ult1,
            ind_deco_fin_ult1,ind_deme_fin_ult1,ind_dela_fin_ult1,ind_ecue_fin_ult1,ind_fond_fin_ult1,ind_hip_fin_ult1,ind_plan_fin_ult1,
            ind_pres_fin_ult1,ind_reca_fin_ult1,ind_tjcr_fin_ult1,ind_valo_fin_ult1,ind_viv_fin_ult1,ind_nomina_ult1,ind_nom_pens_ult1,ind_recibo_ult1,
            
            ind_ahor_fin_ult1_added,ind_aval_fin_ult1_added,ind_cco_fin_ult1_added,ind_cder_fin_ult1_added,ind_cno_fin_ult1_added,ind_ctju_fin_ult1_added,ind_ctma_fin_ult1_added,ind_ctop_fin_ult1_added,ind_ctpp_fin_ult1_added,
            ind_deco_fin_ult1_added,ind_deme_fin_ult1_added,ind_dela_fin_ult1_added,ind_ecue_fin_ult1_added,ind_fond_fin_ult1_added,ind_hip_fin_ult1_added,ind_plan_fin_ult1_added,
            ind_pres_fin_ult1_added,ind_reca_fin_ult1_added,ind_tjcr_fin_ult1_added,ind_valo_fin_ult1_added,ind_viv_fin_ult1_added,ind_nomina_ult1_added,ind_nom_pens_ult1_added,ind_recibo_ult1_added,
            
            ind_ahor_fin_ult1_removed,ind_aval_fin_ult1_removed,ind_cco_fin_ult1_removed,ind_cder_fin_ult1_removed,ind_cno_fin_ult1_removed,ind_ctju_fin_ult1_removed,ind_ctma_fin_ult1_removed,ind_ctop_fin_ult1_removed,ind_ctpp_fin_ult1_removed,
            ind_deco_fin_ult1_removed,ind_deme_fin_ult1_removed,ind_dela_fin_ult1_removed,ind_ecue_fin_ult1_removed,ind_fond_fin_ult1_removed,ind_hip_fin_ult1_removed,ind_plan_fin_ult1_removed,
            ind_pres_fin_ult1_removed,ind_reca_fin_ult1_removed,ind_tjcr_fin_ult1_removed,ind_valo_fin_ult1_removed,ind_viv_fin_ult1_removed,ind_nomina_ult1_removed,ind_nom_pens_ult1_removed,ind_recibo_ult1_removed,
            
            num_products_added,num_products_removed,
            
            num_rare,num_particular,num_deposits,num_finance,num_credit,num_home,num_work,num_pension,num_modern,num_products
            
            ]

        outfile.write(",".join([str(e) for e in v]) + "\n")

        # Current row becomes the previous row
        ppr = pr
        pr = r

# Create all necessary data
def create_all():
    create_preprocessed()
    create_targets()
    create_lasts()
    
def main():
	create_all()
	
if __name__=='__main__':
	main()
    
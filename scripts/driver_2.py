# -*- coding: utf-8 -*-
import os
import papermill as pm
import yaml
import pandas as pd
import numpy as np

def make_dirs(data_directory, project_name):
    directories = ['error', 'results', 'train_test']
    project_directory = os.path.join(data_directory, project_name.replace(' ', '_').lower())
    os.mkdir(project_directory)
    for directory in directories:
        os.mkdir(os.path.join(data_directory,project_name,directory)) 
    return project_directory

def get_data(project_name, project_directory, project_dict, lookback):
    name = '{}_{}.ipynb'.format(project_name,'raw')
    raw_data_path = os.path.join(project_directory,name)
    pm.execute_notebook(
                   r'notebooks/get_data.ipynb',
                   raw_data_path,
                   parameters = dict(project_dict=project_dict, lookback=lookback, project_name=project_name)
                )
    return raw_data_path
    
def get_raw_data(raw_data_path):
     nb = pm.read_notebook(raw_data_path)
     df = nb.dataframe
     raw_data = df[df['name']=='raw_data']['value'].values[0]
     return raw_data
 
def split_data(raw_data, freq, project_directory, project_name):
    #put data in data frame and set index
    df = pd.DataFrame(raw_data)
    df['index'] = pd.to_datetime(df['index'])
    df.set_index('index', inplace = True)
    
    #group data by given frequency
    grouped = df.groupby(pd.Grouper(freq = freq))
    
    #create a dictionary of train/test sets by grouping frequency
    grouped_dict = {}
    for group, data in grouped:
        train = df.drop(index = data.index)
        test = data
        train_test ={}
        for d,name in zip([train, test], ['train', 'test']):
            data ={}
            for column in d.columns:
                data.update({column:list(d[column].values)})
            data.update({'index':[str(x) for x in d.index.values]})
            train_test.update({name:data})
        grouped_dict.update({group:train_test})
    return grouped_dict

def save_grouped_data(grouped_dict,project_name,project_directory):
    #save the split data into individual notebooks
    for k,v in grouped_dict.items():
        file_name = '{}_{}'.format(project_name, str(k).replace(' ', '_').replace(':','_'))
        relative_path = r'train_test\{}.ipynb'.format(file_name)
        abs_path = os.path.join(str(project_directory),relative_path)
        pm.execute_notebook(
           r'notebooks/save_data.ipynb',
           abs_path,
           parameters = dict(group=str(k), data=v)
        )

def get_split_save_data(data_directory,project_name, project_dict, lookback =700, freq = 'Y'):
    project_directory = make_dirs(data_directory, project_name)
    raw_data_path = get_data(project_name, project_directory, project_dict, lookback)
    raw_data = get_raw_data(raw_data_path)
    grouped_dict = split_data(raw_data, freq, project_directory, project_name)
    save_grouped_data(grouped_dict,project_name,project_directory)
    return project_directory
   
def optimize(file_name,train,test, maxiter = 3, x0 = [1,5,2],min_fb_tdg = float('-inf')):
   pm.execute_notebook(
           'notebooks/optimize.ipynb',
           file_name,
           parameters = dict(train=train, test=test, maxiter=maxiter, x0=x0,min_fb_tdg=min_fb_tdg)
        )
       
def optimize_split_data(project_directory, min_fb_tdg ='-inf', file_name_extension= '',maxiter = 3, x0 = [1,5,2]):
    nbs = pm.read_notebooks(os.path.join(project_directory,r'train_test'))
    df = nbs.dataframe
    data = df[df['name']=='data']
    grouped = data.groupby('filename')
    i = 0
    for g,v in grouped: 
        name = g.split('.')[0].replace('-','_')
        train = v['value'].values[0]['train']
        test = v['value'].values[0]['test']
        file_name = os.path.join(project_directory,r'results/{}_optimized{}.ipynb'.format(name,file_name_extension))
        
        optimize(file_name,train=train, test=test, maxiter = maxiter, x0 = x0,min_fb_tdg = min_fb_tdg)
        #get weighted average of x0 to start with for next batch
        nb = pm.read_notebook(file_name)
        nb_df = nb.dataframe
        x = nb_df[nb_df['name']=='x']['value'].values[0]
        x0 = list((np.array(x0) * i + np.array(x))/(i+1))
        i+=1


def calculate_error(file_name, b, train, test, min_fb_tdg='-inf'):
    pm.execute_notebook(
           'notebooks/error.ipynb',
           file_name,
           parameters = dict(file_name=file_name, train=train, test=test,min_fb_tdg=min_fb_tdg)
        )
    
def run_error_on_split_data(project_directory):
    results_directory = os.path.join(project_directory, 'results')
    nbs = pm.read_notebooks(results_directory)
    nb_df = nbs.dataframe
    grouped = nb_df.groupby('filename')
    for g,v in grouped:
        name = g.split('.')[0]
        train = v[v['name']=='train']['value'].values[0]
        test =  v[v['name']=='test']['value'].values[0]
        b = v[v['name']=='x']['value'].values[0]
        min_fb_tdg = v[v['name']=='min_fb_tdg']['value'].values[0]
        file_name = os.path.join(project_directory,'error/{}_error.ipynb'.format(name))
        
        calculate_error(file_name, b, train, test, min_fb_tdg=min_fb_tdg)

    
def run_validation(project_directory,file_name):
    error_directory = os.path.join(project_directory,'error')
    pm.execute_notebook(
           'notebooks/validation.ipynb',
           file_name,
           parameters = dict(error_directory=error_directory)
        )
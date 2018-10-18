# -*- coding: utf-8 -*-

import papermill as pm
import glob

project = 'the_dalles'
maxiter = 75
bootstraps = 1000
lookback = 50000

pm.execute_notebook(
   'get_data.ipynb',
   '{}/data/{}_raw.ipynb'.format(project,project),
   parameters = dict(project=project, lookback=lookback)
)


pm.execute_notebook(
   'split_data.ipynb',
   '{}/data/{}_grouped.ipynb'.format(project,project),
   parameters = dict(project=project, freq='Y')
)


BASE_DATA_DIR = '/data/train_test/'
data_dir = '{}{}*.ipynb'.format(project, BASE_DATA_DIR)
data_files = glob.glob(data_dir)
len_files = len(data_files)


for file in data_files:
    nb = pm.read_notebook(file)
    dic = nb.dataframe.iloc[2,1]
    train = dic['train']
    test = dic['test']
    for i in range(int(bootstraps/len_files)):
        file_name = file.split('\\')[-1].split('.')[0]+'_run_'+str(i)
        nb_name = '{}/data/results/{}'.format(project, file_name+'.ipynb')
        pm.execute_notebook(
           'optimize.ipynb',
           nb_name,
           parameters = dict(dic = train, sample = True, maxiter = maxiter)
        )
        nb = pm.read_notebook(nb_name)
        b1 = nb.dataframe[nb.dataframe['name']=='b1']['value'].values[0]
        b2 = nb.dataframe[nb.dataframe['name']=='b2']['value'].values[0]
        b3 = nb.dataframe[nb.dataframe['name']=='b3']['value'].values[0]
        b = [b1,b2,b3]
    
        pm.execute_notebook(
           'error.ipynb',
           '{}/data/error/{}'.format(project, file_name+'_error.ipynb'),
           parameters = dict(test = test, b = b,)
           )



pm.execute_notebook(
        'validation.ipynb',
         '{}/{}_validation.ipynb'.format(project,project),
         parameters = dict(project=project)
        )


nb = pm.read_notebook('{}/data/raw.ipynb'.format(project))
nb_df = nb.dataframe
dic = nb_df.iloc[2,1]
file_name ='{}/{}_optimize.ipynb'.format(project,project)
pm.execute_notebook(
           'optimize.ipynb',
           file_name,
           parameters = dict(dic = dic, sample = False, maxiter = 100)
        )



nb = pm.read_notebook(file_name)
b1 = nb.dataframe[nb.dataframe['name']=='b1']['value'].values[0]
b2 = nb.dataframe[nb.dataframe['name']=='b2']['value'].values[0]
b3 = nb.dataframe[nb.dataframe['name']=='b3']['value'].values[0]
b = [b1,b2,b3]

pm.execute_notebook(
   'error.ipynb',
   '{}/{}_test.ipynb'.format(project, project),
   parameters = dict(test = dic, b = b,)
   )

pm.execute_notebook(
        'validation.ipynb',
         '{}/{}_validation.ipynb'.format(project,project),
         parameters = dict(project=project)
        )



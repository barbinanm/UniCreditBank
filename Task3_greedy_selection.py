import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqnt
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split,GroupShuffleSplit, cross_val_score, StratifiedShuffleSplit, train_test_split

from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
import pickle
import os
import sys

import warnings
warnings.simplefilter(action='ignore')
warnings.filterwarnings(action='ignore')
# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)


def GreedySelector(X, y,  model, n_folds=5, random_state=84, max_n_features=None, improvement=0, 
                     metric=None, group_field=None, predict='predict_proba',
                     X_test=None, y_test=None, splitter=None,
                     n_jobs=-1, start_features=None, higher_is_better=True,
                    path=None, name=None) -> "fitted_model":
    
    """ 
        X - обучающая выборка. Должна содеражать все признаки, а также поле, по которому нужно группировать данные
            (см. ниже group_field), и начальный(-е) признакам(-и) start_features (см. ниже). Не должна содержать целевую переменную!
            
        y - целевой вектор обучающей выборки.
            
        start_features - здесь лежат уже отобранные признаки. Если нет отобранных признаков,
            то тогда можно не указывать или подать пустой список.
            Если уже отобрали признаки, то в start_features их нужно указать списком (или просто одной строкой,
            если это один признак)
            и в X должны быть эти признаки!
            
        n_folds - число фолдов, на которые будет разбиваться обучающая выборка при кросс-валидации.
        
        random_state - псевдослучайное число/seed. Используется для фиксирования модели, типа разбиения на фолды,
            в общем, для репликации результата после нового запуска.
        
        max_n_features - максимальное число признаков для отбора. Если уже отобрано max_n_features
            признаков, то отбор останавливается.
            
        improvement - минимальное улучшение усредненного качества модели на всех фолдах (по модулю).
            Если после полного одного цикла прогона отбора качество не улучшается (снижается или
            увеличивается в зависимости от параметра higher_is_better (см. ниже)) хотя бы на improvement,
            отбор прекращается.
        
        metric - функция метрики двух аргументов, на улучшение которой и нацелен отбор.
            Первый аргумент - это целевой вектор, второй аргумент - это прогноз модели. Порядок важен!
            По умолчанию считается метрика Джини.
            
        group_field - поле, по которому нужно группировать. То есть, когда происходит разбиение на фолды, наблюдения с
            одним и тем же group_field должны целиком оставаться только в одном фолде.
            В зависимости от наличия или отсутствия group_field выбирается и метод разбиения на фолды.
        
        predict - метод, который вызывается в модели для прогноза. Например, для моделей бинарной классификации из
            библиотеки sklearn (или sklearn'овской обертки, как, например, LGBMClassifier) метод прогноза - 
            model.predict_proba(X)[:,1]. Для прогноза линейной регрессии - model.predict(X). По умолчанию - 
            'predict_proba'
            
        X_test - тестовая выборка. Должна содеражать все признаки, а также поле, по которому нужно группировать данные
            (см. ниже group_field). Не должна содержать целевую переменную!
            Если заданы X_test, y_test (см. ниже) заданы, то кросс-валидации по X не будет! Будет подгонка под
            (X_test, y_test).
            
        y_test - целевой вектор тестовой выборки.
        
        splitter - метод разбиения на фолды выборки X. Если задано поле group_field, то можете выбрать одну из двух
            конфигураций из sklearn.model_selection: ['GroupKFold', 'GroupShuffleSplit']; по умолчанию для такого режима
            стоит 'GroupShuffleSplit'. Если поле group_field не задано, то можете выбрать одну из трех
            конфигураций из sklearn.model_selection: ['KFold', 'StratifiedKFold', 'StratifiedShuffleSplit']; по умолчанию
            для такого режима стоит 'KFold'.
        
        n_jobs - число ядер, использующихся для каждого обучения модели. По умолчанию -1.
        
        start_features - начальный список признаков. Если туда подан хотя бы один признак или набор признаков списком,
            то отбор стартует от этого набора и пытается улучшать качество модели (эти признаки исключаются из общего
            списка признаков), обученной на том наборе признаков.
        
        path - путь, куда записывать pickle лучшей отобранной модели. По умолчанию запись идет в ту же папку, откуда
            запускали отбор.
        
        name - имя pickle модели. По умолчанию 'best_model'.       
        
        """
     
    def model_0(base_model):
        clf = base_model
        params = clf.get_params()
        try:
            if params['random_state'] is None:
                params['random_state'] = random_state
        except:
            pass
        try:
            params['n_jobs'] = n_jobs
        except:
            pass
        clf.set_params(**params)
        return clf
  
    
    def metric_h(y_true, y_pred):
        return 2*roc_auc_score(y_true, y_pred) - 1
        
    def start_metrics_on_folds(higher_is_better):
        if higher_is_better:
            folds_metric = np.zeros(n_folds)            
        else:
            folds_metric = np.ones(n_folds)*999999
                        
        return folds_metric
    
    def fitted_model(X, y, X_test, y_test, base_model, group_field):
        clf = base_model
        
        feat_list = X.columns.tolist()
        
        if group_field:            
            feat_list.remove(group_field)    
            
#         if not((X_test is None) or (y_test is None)):
#             X_test = X_test[feat_list]
            
        if (X_test is None) or (y_test is None):
            clf.fit(X[feat_list], y)
        else:
            try:
                clf.fit(X, y,
                    eval_set=[(X[feat_list], y), (X_test[feat_list], y_test)],
                    eval_metric='auc',
                    verbose=False)
            except:
                clf.fit(X[feat_list], y)
                
        return clf       
        
     
    def metrics_on_folds_long(X, y, X_test, y_test, base_model, metric, n_folds, n_jobs, random_state,
                              predict, splitter, group_field):
        """ 
        Вот здесь считается скор на n_folds фолдах обучающей выборки X или на тестовой выборке X_test при наличии последней
        """
        feat_list = X.columns.tolist()
        if group_field:
            feat_list.remove(group_field)       
            
        if not((X_test is None) or (y_test is None)):
            X_test = X_test[X.columns.tolist()]
                        
#         metrics_on_folds = []
        
        if n_folds > 1 and ((X_test is None) or (y_test is None)):
            if group_field:
#                 cv = GroupKFold(n_splits=n_folds).split(X[feat_list], y, X[group_field])
                if not splitter:
                    cv = GroupShuffleSplit(n_splits=n_folds, random_state=random_state, test_size=1/n_folds)
#                     .split(X[feat_list], y, X[group_field])
                else:
                    cv = GroupKFold(n_splits=n_folds)
#                 .split(X[feat_list], y, X[group_field])
    
            else:
                if not splitter:
                    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
#                     .split(X[feat_list], y)
                elif splitter == 'StratifiedKFold':
                    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                else:
                    cv = StratifiedShuffleSplit(n_splits=n_folds, shuffle=True, test_size=1/n_folds,
                                                random_state=random_state)
                    
            clf = base_model
            
            def g_scoring(mod, x_, y_):
                if predict == 'predict_proba':
                    y_pred = mod.predict_proba(x_)[:,1]
                else:
                    y_pred = mod.predict(x_)                    
                return metric(y_, y_pred)
            
            if group_field:
                metrics_on_folds = cross_val_score(clf, X[feat_list], y, groups=X[group_field],
                                                   scoring=g_scoring, cv=cv, n_jobs=n_jobs)
            else:
                metrics_on_folds = cross_val_score(clf, X[feat_list], y,                                                   scoring=g_scoring, cv=cv, n_jobs=n_jobs)
                    

            
        elif n_folds == 1 and ((X_test is None) or (y_test is None)):
            x_tr = X[feat_list]
            y_tr = y
            clf = base_model
            clf.fit(x_tr, y_tr)
            if predict == 'predict_proba':
                y_pred = clf.predict_proba(x_tr)[:,1]
            else:
                y_pred = clf.predict(x_tr)
            metric_1_fold  = metric(y_tr, y_pred)
#             metrics_on_folds.append(metric_1_fold)
            metrics_on_folds = np.array([metric_1_fold])
            
        elif not((X_test is None) or (y_test is None)):
            x_tr = X[feat_list]
            y_tr = y
            x_te = X_test[feat_list]
            y_te = y_test
            clf = fitted_model(x_tr, y_tr, x_te, y_te, base_model)
            if predict == 'predict_proba':
                y_pred = clf.predict_proba(x_te)[:,1]
            else:
                y_pred = clf.predict(x_te)
            metric_1_fold  = metric(y_te, y_pred)
#             metrics_on_folds.append(metric_1_fold)
            metrics_on_folds = np.array([metric_1_fold])
      
        return metrics_on_folds
    
    """
    Погнали
    """
    if name is None:
        name = 'best_model'
    
    if max_n_features is None:
        max_n_features = X.shape[1]
    
    base_model = model_0(model)
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
        
    if not(X_test is None or y_test is None):
        n_folds = 1
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
    
    if metric is None:
        metric = metric_h
        
#     check_eval(base_model, X, y, group_field)   
           
    if not start_features:
        feature_list = X.columns.tolist()
        if group_field:
            feature_list.remove(group_field)
        feature_remains = feature_list.copy()
        metrics_folds_0 = start_metrics_on_folds(higher_is_better)
        selected_features = []

    else:
        if type(start_features) == type('a'):
            start_features = [start_features]
        feature_list = X.drop(columns=start_features).columns.tolist()
        if group_field:
            feature_list.remove(group_field)
        feature_remains = feature_list.copy()
        if group_field:
            metrics_folds_0 = metrics_on_folds_long(X[[group_field] + start_features], y, X_test, y_test, base_model,
                                                        metric, n_folds, n_jobs, random_state, predict, splitter, group_field)
            
        else:
            metrics_folds_0 = metrics_on_folds_long(X[start_features], y, X_test, y_test, base_model,
                                                        metric, n_folds, n_jobs, random_state, predict, splitter, group_field=None)
            
        best_estimator = fitted_model(X[start_features], y, X_test, y_test, base_model, group_field=None)
        selected_features = list(start_features).copy()
         
    mean_metric_0 = metrics_folds_0.mean()
    best_metrics_folds = metrics_folds_0
    best_mean_metric = mean_metric_0
    print('Всего признаков -', len(feature_list))
     
    for outer in tqnt(feature_list):

        print('Уже отобрано признаков -', len(selected_features), '\n')
        last_best_mean_metric = best_mean_metric
        last_best_metrics_folds = best_metrics_folds
        improve_table = pd.DataFrame(columns=['features', 'mean_metric', 'folds_metrics', 'delta'])
        if len(selected_features) < max_n_features:
                
            for inner in tqnt(feature_remains):
                current_features = selected_features.copy()
                current_features.append(inner)
                
                if group_field:
                    current_metrics_folds = metrics_on_folds_long(X[[group_field] + current_features], y, X_test, y_test, base_model,
                                                            metric, n_folds, n_jobs, random_state, predict, splitter, group_field)

                else:
                    current_metrics_folds = metrics_on_folds_long(X[current_features], y, X_test, y_test, base_model,
                                                            metric, n_folds, n_jobs, random_state, predict, splitter, group_field=None)
                
#                 current_metrics_folds = metrics_on_folds_long(X[current_features], y, X_test, y_test, base_model,
#                                                                  metric, n_folds, n_jobs, random_state)
                current_mean_metric = current_metrics_folds.mean()            
                difference_metrics_folds = current_metrics_folds - last_best_metrics_folds
                difference_mean_metric = current_mean_metric - last_best_mean_metric

                if higher_is_better:
    #                 if (len(selected_features) > max_n_features) and (difference_mean_metric < 0):
    #                     feature_remains.remove(inner)                    
                    num_good_folds = len([i for i in difference_metrics_folds if i > 0])
                    if (num_good_folds > n_folds//2) and (difference_mean_metric > improvement):
                        best_mean_metric = current_mean_metric
                        best_metrics_folds = current_metrics_folds
                        best_features = current_features.copy()
                        delta = difference_mean_metric
                        improve_table.loc[improve_table.shape[0]] = [best_features, best_mean_metric, best_metrics_folds, delta]
                        del best_features
                else:
    #                 if (len(selected_features) > max_n_features) and (difference_mean_metric > 0):
    #                     feature_remains.remove(inner) 
                    num_good_folds = len([i for i in difference_metrics_folds if i < 0])
                    if (num_good_folds > n_folds//2) and (-difference_mean_metric > improvement):
                        best_mean_metric = current_mean_metric
                        best_metrics_folds = current_metrics_folds
                        best_features = current_features.copy()
                        delta = -difference_mean_metric
                        improve_table.loc[improve_table.shape[0]] = [best_features, best_mean_metric, best_metrics_folds, delta]
                        del best_features

            if improve_table.shape[0] > 0:
                best_features = improve_table.sort_values(by='delta', ascending=False).iloc[0, :]['features']
                best_mean_metric = improve_table.sort_values(by='delta', ascending=False).iloc[0, :]['mean_metric']
                best_metrics_folds = np.array(improve_table.sort_values(by='delta', ascending=False).iloc[0, :]['folds_metrics'])
                best_add = best_features[-1]
            else:
                best_features = selected_features

            del improve_table

            if len(list(set(best_features) - set(selected_features))) > 0 and len(best_features) <= max_n_features:
                selected_features = best_features
                best_estimator = fitted_model(X[selected_features], y, X_test, y_test, base_model, group_field=None)
                refit_metric = metrics_on_folds_long(X=X[selected_features],
                                                     y=y,
                                                     X_test=X_test,
                                                     y_test=y_test,
                                                     base_model=base_model,
                                                     metric=metric,
                                                     n_folds=1,
                                                     random_state=random_state,
                                                     n_jobs=n_jobs,
                                                    predict=predict,
                                                     splitter=splitter,
                                                    group_field=None)[0]

                if (X_test is None) or (y_test is None):
                    print('=============================================================', '\n',
                        'Отобранные признаки:', selected_features, '\n',
                         'Метрика качества на обучающей выборке:', refit_metric, '\n',
                         '=============================================================')

                else:                
                    print('=============================================================', '\n',
                        'Отобранные признаки:', selected_features, '\n',
                        'Метрика качества на тестовой выборке:', refit_metric, '\n',   
                         '=============================================================')

                feature_remains.remove(best_add)
                if path is not None:
                    with open(path + name, 'wb') as f:
                        pickle.dump(best_estimator, f)
                else:
                    with open(name, 'wb') as f:
                        pickle.dump(best_estimator, f)

            else:
                print('Метрика качества больше не улучшается. Все посчиталось. Поздравляю!')
                break
        
        else:
            print(f'{max_n_features} уже отобралось. Поздравляю!')
            break
    best_estimator = fitted_model(X[selected_features], y, X_test, y_test, base_model, group_field=None)
           
    if path is not None:
        with open(path + name, 'wb') as f:
            pickle.dump(best_estimator, f)
    else:
        with open(name, 'wb') as f:
            pickle.dump(best_estimator, f)    

    return best_estimator


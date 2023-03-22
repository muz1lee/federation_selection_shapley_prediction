import os
import sys
import torch
import copy
import pandas as pd
from global_tools import save_pkl, append_in_dict
from utils.log_tools import add_scalars, add_best_scalars, fprint, close_tb_recorder

class BasicFramework(object):
    best_info = {}
    args = None
    COMPARED_METRIC_NAME = "test_acc"
    
    def save_best_to_csv(self, best_info):
        """save best metric and the run parameters locally and globally(in dirpath/metric.csv) 

        :param dict best_info: dict of the best metrics.
        """
        run_path = self.args.base_dir # the folder path of this run.
        local_path = os.path.join(run_path, "metric.csv")
        global_path = os.path.join(os.path.dirname(run_path), "metric.csv")
        new_data = {**vars(self.args), **best_info}
        new_data = {k: [v] for k,v in new_data.items()}
        df = pd.DataFrame(new_data)
        df.to_csv(local_path,index=False,sep=',') # local save
        
        if os.path.exists(global_path):
            df_global = pd.read_csv(global_path)
            if set(df_global.columns).issubset(set(df.columns)):
                df_global = pd.concat([df, df_global])
            else:
                df_global = pd.concat([df_global, df])
        else:
            df_global = df
        df_global.to_csv(global_path,index=False,sep=',') # global save

    def save_to_file(self, name, results):
        save_pkl(os.path.join(self.base_dir, "{}.pkl".format(name)), results)
        best_info_remove_model = {k:v for k,v in self.best_info.items() if k != 'model'}
        fprint("Save params{}".format(vars(self.args)))
        self.save_best_to_csv(best_info_remove_model)
        add_best_scalars(vars(self.args), best_info_remove_model)
        close_tb_recorder()

    def set_start_epoch(self, epoch):
        func_info = "{}->{}".format(BasicFramework.__qualname__, sys._getframe().f_code.co_name)
        fprint(func_info, level='DEBUG')
        self.start_epoch = epoch

    def record_log(self, history, results_cat, weight_quality, glob=True, is_best=False, tb_weight_quality=False):
        history = append_in_dict(history, results_cat)
        if weight_quality is not None:
            if "weight_quality" not in history:
                history["weight_quality"] = {}
            history['weight_quality'] = append_in_dict(history['weight_quality'], weight_quality)

        if glob:
            epoch = results_cat['epoch']
            metrics = {k:v for k,v in results_cat.items() if k != 'epoch'}
            add_scalars(epoch, metrics)

            if weight_quality is not None:
                tag_weight_quality = {k.replace('_','/'):v for k,v in weight_quality.items()}
                add_scalars(epoch, tag_weight_quality)

            if is_best:
                for key, value in results_cat.items():
                    self.best_info[ key] = value

        return history
    
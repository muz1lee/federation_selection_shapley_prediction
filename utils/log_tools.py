import os
import datetime
import logging
import traceback
import shutil
from torch.utils.tensorboard import SummaryWriter


tb_recorder = None

def generate_log_dir(path, is_use_tb=True, ind_sim=None, has_timestamp=True):
    """
    initialize wandb enviro
    :param project_name: name
    :param hyper_params: list of params
    :return:
    """
    base_dir = path

    if ind_sim is None:
        # when base_dir has existed, it means that some special parameters did not emerge in the log dir. So we add timestamp to prevent overwrite existing log.
        if os.path.exists(base_dir) and has_timestamp:
            timestamp = datetime.datetime.now().strftime('%m%d-%H%M%S-%f')
            base_dir = '%s_%s' % (base_dir, timestamp)
    else:
        base_dir = os.path.join(base_dir, ind_sim)
    os.makedirs(base_dir, exist_ok=True)
    
    close_logger()
    set_logger(base_dir)
    if is_use_tb:
        close_tb_recorder()
        init_tb_recorder(path=base_dir)

    return base_dir

## logger save training time information and error.
logger = None
def set_logger(root):
    global logger
    logger = logging.getLogger('autoLog')
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s: %(message)s')

    path = os.path.join(root, 'output.log')
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def get_logger():
    global logger
    return logger

def close_logger():
    global logger
    if logger is not None:
        logger.handlers.clear()

# tensorboard recorder.
def init_tb_recorder(path):
    global tb_recorder
    tb_recorder = SummaryWriter(log_dir=path)
    
def add_scalars(epoch, results:dict):
    global tb_recorder
    for metric_name, value in results.items():
        if '/' not in metric_name:
            if 'global' in metric_name:
                tag_name = 'global_metrics/'
            else:
                tag_name = 'metrics/'
        else:
            tag_name = ""
        tb_recorder.add_scalar(tag_name + metric_name, value, global_step=epoch)
    tb_recorder.flush()

def add_best_scalars(params, best_results):
    global tb_recorder
    metric_dict = {"hparam/best_{}".format(key):value for key, value in best_results.items()}
    tb_recorder.add_hparams(params, metric_dict)

def close_tb_recorder():
    global tb_recorder
    if tb_recorder is not None:
        tb_recorder.close()


def fprint(m, level='INFO'):
    global logger
    if level == 'INFO':
        logger.info(m)
    elif level == 'DEBUG':
        logger.debug(m)
    elif level == 'ERROR':
        logger.error(m)
    elif level == 'CRITICAL':
        logger.critical(m)

class CatchExcept:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type:
            if get_logger() is None:
                print(traceback.format_exc(), level="ERROR")
            else:
                fprint(traceback.format_exc(), level="ERROR")
            close_tb_recorder()
            run_path = os.path.dirname(get_logger().handlers[0].baseFilename)
            close_logger()
            exp_path = os.path.dirname(run_path) # expx_xx
            # if run is fail, please mv it to bin/ directory
            if "KeyboardInterrupt" in traceback.format_exc():
                mid_dir = "KeyboardInterrupt"
            else:
                mid_dir = "bin"
            new_path = os.path.join(exp_path, mid_dir, os.path.basename(run_path)) # xx_xx_xx_x
            if 'exp' in exp_path:
                shutil.move(run_path, new_path)

        return True
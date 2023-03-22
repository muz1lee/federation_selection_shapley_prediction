import os
import re
import sys
from shutil import copy2

project_path = os.path.abspath('.')
all_code_file_paths = set()

def check_modules(main_path:str):
    global all_code_file_paths
    all_code_file_paths.add(main_path)
    mods = sys.modules
    for k, v in mods.items():
        try:
            if project_path in v.__spec__.origin:
                all_code_file_paths.add(v.__spec__.origin)
        except:
            pass

def find_longest_public_str(strs: list):
    public_str = ''
    for i in range(len(strs[0])):
        cur_str = strs[0][i]
        flag = True
        for s in strs:
            if s[i] != cur_str:
                flag = False
                break
        if flag:
            public_str += cur_str
        else:
            break
    return public_str



def copy_least_files(to_path):
    global all_code_file_paths
    all_code_file_paths = list(all_code_file_paths)
    public_str = find_longest_public_str(all_code_file_paths)
    for p in all_code_file_paths:
        p_new = os.path.join(to_path, p.replace(public_str, ''))
        dir_new = os.path.dirname(p_new)
        if not os.path.exists(dir_new):
            os.makedirs(dir_new)
        copy2(src=p, dst=p_new)


def backup_code(src_dir, to_dir):
    ignore_dir_keywords = ['.vscode', '__pycache__', '.git', 'wandb', '.ipynb_checkpoints', 
                           '.pdf', '.csv','.gif','.npy']
    pattern_list = [re.compile(r'{}'.format(keyword)) for keyword in ignore_dir_keywords]
    
    if src_dir[-1] != '/':
        src_dir += '/'

    def is_ignore(path, pattern_list):
        for pattern in pattern_list:
            if pattern.search(path) is not None:
                return True
        return False
    def path_generater(src_dir, pattern_list):
        for root, dir, files in os.walk(src_dir):
            for file in files:
                if not is_ignore(os.path.join(root, file), pattern_list):
                    yield os.path.join(root, file)

    for path in path_generater(src_dir, pattern_list):
        if os.path.getsize(path) / float(1024*1024) >= 1:
            print(f"Warning {path_new} > 1 MB")
        path_new = os.path.join(to_dir, path.replace(src_dir, ''))
        dir_new = os.path.dirname(path_new)
        if not os.path.exists(dir_new):
            os.makedirs(dir_new)
        # print(path_new)
        copy2(src=path, dst=path_new)

    # backup_code("/home/xuyc/projects/fedvar_inc", '')
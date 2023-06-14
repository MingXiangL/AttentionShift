import os
import shutil
from tqdm import tqdm as tqdm
import pdb
import time

def gettime():
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

def make_bak_path(path_sup, file_name):
    succeed = False
    i = 1
    time_token = gettime()
    path_bak = os.path.join(path_sup, file_name + f'-{time_token}.bak')
    
    while not succeed:
        if not os.path.exists(path_bak):
            os.makedirs(path_bak)
            succeed = True
            print(path_bak+ 'maked!!')
        else:
            i += 1
            path_bak = os.path.join(path_sup, file_name + f'-{time_token}.bak{i}')
            print(path_bak)

    print(f'The path for baking is {path_bak}')
    return path_bak

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    path_sup = os.path.dirname(path)
    file_name = os.path.basename(path)
    file_list = os.listdir(path)
    path_bak = make_bak_path(path_sup, file_name)
    except_list = ['data', 'output', 'work_dirs', 'nohup.out']

    for file in tqdm(file_list):
        if file in except_list:
            continue
        
        file_abs_path = os.path.join(path, file)
        if os.path.isfile(file_abs_path):
            print(f'copying file {file_abs_path} to {path_bak}')
            shutil.copy(file_abs_path, path_bak)
        else:
            print(f'copying file {file_abs_path} to {os.path.join(path_bak, file)}')
            shutil.copytree(file_abs_path, os.path.join(path_bak, file))


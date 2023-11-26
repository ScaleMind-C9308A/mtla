import os, sys
import json
import shutil

def folder_setup(args):
    run_dir = os.getcwd() + "/runs"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    setting_dir = run_dir + f"/{args.ds}_{args.method}"
    if not os.path.exists(setting_dir):
        os.mkdir(setting_dir)
    
    exp_cnt = len(os.listdir(setting_dir))
    if exp_cnt == 0:
        exp_dir = setting_dir + f"/exp_{exp_cnt}"
    else:
        old_exp_dir = setting_dir + f"/exp_{exp_cnt-1}"
        if len(os.listdir(old_exp_dir)) > 0:
            cfg_path = old_exp_dir + "/config.json"
            if not os.path.exists(cfg_path):
                shutil.rmtree(old_exp_dir)
                exp_dir = old_exp_dir
            else:
                f = open(cfg_path, 'r')
                data = json.load(f) 
                if data['test']:
                    shutil.rmtree(old_exp_dir)
                    exp_dir = old_exp_dir
                else:
                    exp_dir = setting_dir + f"/exp_{exp_cnt}"
        else:
            exp_dir = old_exp_dir
    
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    return exp_dir


def save_cfg(args, exp_dir = None):
    config_dict = vars(args)

    if not exp_dir:
        raise ValueError('exp_dir cannot be None')
    else:
        path = exp_dir + "/config.json"

        with open(path, "w") as outfile: 
            json.dump(config_dict, outfile)

def save_json(dct, path):
    with open(path, "w") as outfile: 
        json.dump(dct, outfile)
import os, sys

def folder_setup(args):
    run_dir = os.getcwd() + "/runs"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    setting_dir = run_dir + f"/{args.ds}_{args.method}"
    if not os.path.exists(setting_dir):
        os.mkdir(setting_dir)
    
    exp_dir = setting_dir + f"/exp_{len(os.listdir(setting_dir))}"
    os.mkdir(exp_dir)

    return exp_dir
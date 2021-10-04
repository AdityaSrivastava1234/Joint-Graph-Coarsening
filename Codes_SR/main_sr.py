# %%
def sitaram():
    return "Siiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiitaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaam"
print(sitaram())

# %%
# *********************** Importing Essential Libraries *************************
import os
import argparse
import torch
import numpy as np
import random

import utilis_sr
import models_sr
import train_procedures_sr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device name :- "+str(device))

print(sitaram())

# ************** Initializing Argparser ********************
parser = argparse.ArgumentParser(description='Standard ML hyper-params')
parser.add_argument('--noe',type=int,default=201)
parser.add_argument('--alpha',type=float,default=0.01)
parser.add_argument('--wd',type=float,default=0.0005)
parser.add_argument('--runs', type=int, default=21)
parser.add_argument('--normalize_features',type=bool,default=True)
parser.add_argument('--save_best',type=bool,default=True)
parser.add_argument('--jobid',type=str,default=None)
parser.add_argument('--rand_seed',type=int,default=-1)
parser.add_argument('--only_deterministic',type=bool,default=False)
parser.add_argument('--mn',type=str,default='Model_sd_SR')
parser.add_argument('--save_plts_evry_run',type=bool,default=False)
parser.add_argument('--model_type',type=str,default='APPNP_SR')
parser.add_argument('--coarsening_ratio', type=float, default=0.5)
parser.add_argument('--coarsening_method', type=str, default='variation_edges')
parser.add_argument('--dataset', type=str, default='cora_SR')
parser.add_argument('--experiment', type=str, default='fixed_SR') #'fixed_SR', 'random_SR', 'few_SR'
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha_appnp', type=float, default=0.1)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--prnt_mean_std',type=bool,default=True)

args = parser.parse_args()
print(vars(args))

print(sitaram())

# **************** Some Relevant Paths **************************
root_path = "/home/mech/btech/me1180666/scratch/BTP_SR/"
data_path = root_path+"Datasets_SR/"
model_path = root_path+"Models_SR/"
save_model_path = model_path+args.jobid
save_plot_path = root_path+"Plots_SR/"+args.jobid
exist_ok_val=False
if args.jobid=='abc_sr':
    exist_ok_val=True
os.makedirs(save_plot_path,exist_ok=exist_ok_val)
os.makedirs(save_model_path,exist_ok=exist_ok_val)

# %%
# ************** Initializing Global Configs Dict ********************
global_configs = {}
global_configs['epochs'] = args.noe
global_configs['learning_rate'] = args.alpha
global_configs['weight_decay'] = args.wd
global_configs['normalize_features'] = args.normalize_features
global_configs['runs'] = args.runs
global_configs['save_best_model'] = args.save_best
global_configs['save_model_path'] = save_model_path + '//' + args.mn +'.pt'
global_configs['save_plot_path'] = save_plot_path
global_configs['save_all_plots'] = args.save_plts_evry_run
global_configs['data_path'] = data_path
global_configs['dataset'] = args.dataset
global_configs['experiment'] = args.experiment
global_configs['coarsening_ratio'] = args.coarsening_ratio
global_configs['coarsening_method'] = args.coarsening_method
global_configs['K'] = args.K
global_configs['print_net_test_info'] = args.prnt_mean_std
if args.rand_seed!=-1:
    global_configs['rand_seed'] = args.rand_seed
else:
    global_configs['rand_seed'] = random.randint(0,(2**32)-1)
    print("Using Random Seed ==> "+str(global_configs['rand_seed']))
print(sitaram())

# %%
# ************* Random Seed to all the devices ****************
torch.backends.cudnn.deterministic = args.only_deterministic
torch.manual_seed(global_configs['rand_seed'])
np.random.seed(global_configs['rand_seed'])
random.seed(global_configs['rand_seed'])
# train_procedures_sr.rand_seed_devices(configs=global_configs)
# single_encoder_train_procedures_sr.rand_seed_devices(configs=global_configs)
# train_utilities_sr.rand_seed_devices(configs=global_configs)
# test_procedures_sr.rand_seed_devices(configs=global_configs)
# utilis_sr.rand_seed_devices(configs=global_configs)
# inference_sr.rand_seed_devices(configs=global_configs)
# data_processing_sr.rand_seed_devices(configs=global_configs)
# models_sr.rand_seed_devices(configs=global_configs)
print(sitaram())

# %%
# **************** Coarsen the Graph SitaRam *************************
global_configs['num_features'], global_configs['num_classes'], global_configs['candidate'], global_configs['C_list'], global_configs['Gc_list'] = utilis_sr.coarsening(configs=global_configs)
print(sitaram())

# %%
# **************** Initializing Model **************************
if args.model_type=='APPNP_SR':
    global_configs['beta'] = args.alpha_appnp
    global_configs['hidden'] = args.hidden
    Model_sr = models_sr.APPNP_SR(configs=global_configs).to(device)
elif args.model_type=='GCN_SR':
    Model_sr = models_sr.GCN_SR(configs=global_configs).to(device)
print(sitaram())

# %%
# ***************** Training Model ***********************
Model_sr = train_procedures_sr.train_Model_end_to_end(Model=Model_sr,configs=global_configs)
print(sitaram())
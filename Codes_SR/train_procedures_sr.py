# %%
# *********************** Importing Essential Libraries *************************
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import numpy as np
import time

import test_procedures_sr
import data_processing_sr
import utilis_sr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def load_Data_and_train_Model_single_run_SR(Model,configs,save_plots=True,want_verbose=True,ret_logger=False):

    logger_dict={}
    logger_dict['Train_Loss'] = []
    logger_dict['Val_Loss'] = []
    logger_dict['Train_Accuracy'] = []
    logger_dict['Val_Accuracy'] = []

    data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = data_processing_sr.load_data(configs)

    data = data.to(device)
    coarsen_features = coarsen_features.to(device)
    coarsen_train_labels = coarsen_train_labels.to(device)
    coarsen_train_mask = coarsen_train_mask.to(device)
    coarsen_val_labels = coarsen_val_labels.to(device)
    coarsen_val_mask = coarsen_val_mask.to(device)
    coarsen_edge = coarsen_edge.to(device)

    train_info_dict = {}
    train_info_dict['labels'] = coarsen_train_labels
    train_info_dict['mask'] = coarsen_train_mask
    train_info_dict['features'] = coarsen_features
    train_info_dict['edges'] = coarsen_edge

    val_info_dict = {}
    val_info_dict['labels'] = coarsen_val_labels
    val_info_dict['mask'] = coarsen_val_mask
    val_info_dict['features'] = coarsen_features
    val_info_dict['edges'] = coarsen_edge
    
    if configs['normalize_features']:
        coarsen_features = F.normalize(coarsen_features, p=1)
        data.x = F.normalize(data.x, p=1)

    Model.reset_parameters()
    optimizer = utilis_sr.make_opt_sr(model=Model,lr=configs['learning_rate'],wd=configs['weight_decay'])

    best_val_loss = float('inf')

    for epoch in range(configs['epochs']):
        start_time = time.time()
        Model.train()

        optimizer.zero_grad()
        out = Model(coarsen_features, coarsen_edge)
        loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
        loss.backward()
        logger_dict['Train_Loss'].append(loss.item())
        optimizer.step()

        logger_dict = test_procedures_sr.test_model_sr(logger=logger_dict,Model=Model,info_dict=train_info_dict,key_wrd='Train')
        logger_dict = test_procedures_sr.test_model_sr(logger=logger_dict,Model=Model,info_dict=val_info_dict,key_wrd='Val')

        if logger_dict['Val_Loss'][-1] < best_val_loss and configs['save_best_model']:
            best_val_loss = logger_dict['Val_Loss'][-1]
            torch.save(Model.state_dict(), configs['save_model_path'])

        # if args.early_stopping > 0 and epoch > args.epochs // 2:
        #     tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
        #     if val_loss > tmp.mean().item():
        #         break

        if want_verbose:
            print('Epoch [{}/{}], Time Taken: {:.2f}s, Train Loss: {:.5f}, Train Accuracy: {:.2f}%, Val Loss: {:.5f}, Val Accuracy: {:.2f}%'.format(epoch+1,configs['epochs'],time.time()-start_time,logger_dict['Train_Loss'][-1],logger_dict['Train_Accuracy'][-1],logger_dict['Val_Loss'][-1],logger_dict['Val_Accuracy'][-1]))

    if configs['save_best_model']:
        Model.load_state_dict(torch.load(configs['save_model_path'],map_location=device))
    
    test_info_dict = {'features':data.x, 'labels':data.y, 'mask':data.test_mask, 'edges':data.edge_index}
    logger_dict = test_procedures_sr.test_model_sr(logger=logger_dict,Model=Model,info_dict=test_info_dict)

    if save_plots:
        utilis_sr.plot_train_val_curve(logger_dict['Train_Loss'],logger_dict['Val_Loss'],configs['epochs'],nm='Loss_Run'+str(configs['curr_run_sr']+1),path=configs['save_plot_path'],is_percent=False)
        utilis_sr.plot_train_val_curve(logger_dict['Train_Accuracy'],logger_dict['Val_Accuracy'],configs['epochs'],nm='Accuracy_Run'+str(configs['curr_run_sr']+1),path=configs['save_plot_path'])

    if want_verbose:
        print("")
        print("Test Loss = "+str(logger_dict['Test_Loss']))
        print("Test Accuracy = "+str(logger_dict['Test_Accuracy']))
        print("")

    if not ret_logger:
        logger_dict=None

    return Model,logger_dict

# %%
def train_Model_end_to_end(Model,configs):
    
    if configs['print_net_test_info']:
        test_accuracies_sr=[]
        test_losses_sr=[]

    for run in range(configs['runs']):
        configs['curr_run_sr']=run
        save_plots=configs['save_all_plots']
        if run==configs['runs']-1:
            save_plots=True
        print("")
        print("Executing Run Number "+str(run+1)+" -->")
        Model,logger = load_Data_and_train_Model_single_run_SR(Model=Model,configs=configs,save_plots=save_plots,ret_logger=configs['print_net_test_info'])

        if configs['print_net_test_info']:
            test_accuracies_sr.append(logger['Test_Accuracy'])
            test_losses_sr.append(logger['Test_Loss'])

    if configs['print_net_test_info']:
        print("")
        print('Avg_Test_Loss: {:.4f}'.format(np.mean(test_losses_sr)), '+/- {:.4f}'.format(np.std(test_losses_sr)))
        print('Avg_Test_Accuracy: {:.4f}%'.format(np.mean(test_accuracies_sr)), '+/- {:.4f}'.format(np.std(test_accuracies_sr)))

    return Model
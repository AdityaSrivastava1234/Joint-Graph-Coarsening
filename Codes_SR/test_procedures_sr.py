import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def test_model_sr(logger,Model,info_dict,key_wrd='Test'):
	Model.eval()
	with torch.no_grad():
		pred = Model(info_dict['features'], info_dict['edges'])
		if key_wrd!='Train':
			loss_sr = F.nll_loss(pred[info_dict['mask']], info_dict['labels'][info_dict['mask']]).item()
	pred = pred.max(1)[1]
	acc_sr = (int(pred[info_dict['mask']].eq(info_dict['labels'][info_dict['mask']]).sum().item()) / int(info_dict['mask'].sum()))*100
	try:
		if key_wrd!='Train':
			logger[key_wrd+'_Loss'].append(loss_sr)
		logger[key_wrd+'_Accuracy'].append(acc_sr)
	except:
		if key_wrd!='Train':
			logger[key_wrd+'_Loss']=loss_sr
		logger[key_wrd+'_Accuracy']=acc_sr
	return logger
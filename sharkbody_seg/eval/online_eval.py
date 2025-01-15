import random
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.segmentation import mean_iou

@torch.inference_mode()
def online_eval(model, dataloader, 
        criterion, device, 
        cfg=None, dtype=torch.float32,
        wandb_run=None):
    '''
    Compute scores on validation dataset during training

    Args:
        model torch.nn.Module : neural network object
        dataloader torch.utils.data.dataloader.DataLoader
        criterion torch.nn.Module: loss function
        device torch.device: device, e.g., cpu or gpu
        cfg dict: Config file with all hyperparameters.
        wandb_run: optional wandb logging object created with wandb.init()
    '''
    model.eval()
    n_val = len(dataloader.dataset)
    total_loss = 0.
    batch_sizes = [] # placeholder. Batch size varies with dataloader.drop_last = False 
    if len(dataloader) == 0.: # Set batch size if val_size == 0
        batch_sizes = [1]
    # Random index of image that will be plotted
    # generates int between 0 (inclusive) and n_val (exclusive)
    plot_img_idx = torch.randint(0, n_val, (1,)) 

    iou_score_list = [] # initialize list

    # iterate over the validation set
    with tqdm(total=n_val, desc='validation.', unit='img', leave=False) as pbar2:
        for i, batch in enumerate(dataloader):
            inputs = batch['image']
            targets = batch['mask']

            batch_sizes.append(inputs.shape[0]) 
            inputs = inputs.to(device=device, dtype=dtype, memory_format=torch.channels_last)
            targets = targets.to(device=device, dtype=dtype)

            # predict the output
            pred = model(inputs)

            loss = criterion(pred, targets) # average loss per img
            total_loss += loss * batch_sizes[i] # compute total loss by multiplying with number of images in batch

            pred = torch.sigmoid(pred)>0.5 # parameter to tune; change to parameter in config (**)
        
            pbar2.update(batch_sizes[i])
            pbar2.set_postfix(**{'val loss/img': loss.cpu().numpy() / float(i+1)})
            
            # iou calculation, log to weights and biases
            iou = mean_iou(pred.to(torch.long), targets.to(torch.long), num_classes = 2, input_format = "index")
            iou_score = iou[1].item() # iou from tensor to scalar
            iou_score_list.append(iou_score) # add to list for this epoch

            # Plot one image in the batch to wandb
            if i == np.floor(plot_img_idx / batch_sizes[0]): # index into batch that contains plot_img_idx
                log_inputs_and_targets_to_wandb = cfg['log_inputs_and_targets_to_wandb'] if 'log_inputs_and_targets_to_wandb' in cfg.keys() else True
                if wandb_run is not None and log_inputs_and_targets_to_wandb:
                    import wandb
                    idx_in_batch = (plot_img_idx - i * batch_sizes[0]).cpu().numpy().item() # index of image in batch inputs[] tensor
                    # Add all in- and output images as a list of images, s.t., epoch slider moves all the same.
                    log_ims_wandb = [wandb.Image(input_img) for input_img in inputs[idx_in_batch].cpu()] # add inputs
                    # log_ims_wandb.append(wandb.Image(targets_mask[idx_in_batch,0].cpu().numpy())) # {0: 'valid', 1: 'masked'}
                    log_ims_wandb.append(wandb.Image(targets[idx_in_batch].cpu())) # add target
                    log_ims_wandb.append(wandb.Image(pred[idx_in_batch].cpu())) # add prediction
                    if 'in_keys' in cfg:
                        in_keys = cfg['in_keys'] + cfg['in_keys_static']
                    else:
                        in_keys = ''
                    wandb_run.log({'inputs('+','.join(in_keys)+'), target-mask, target, pred': 
                                log_ims_wandb}, commit=False)

    # Accumulated IOU
    iou_accum = sum(iou_score_list)/len(iou_score_list) # take iou avg for epoch
     # Log the IoU
    wandb.log({'iou_accum':iou_accum}) # log iou as a tensor  

    # Compute val score
    val_score = total_loss / n_val
    logging.info('val loss/img: {}'.format(val_score))
 
    # Log plots to wandb
    if wandb_run is not None:
        import wandb

        # Optionally log distribution of weights and gradients per layer. This takes ~1sec per log. 
        histograms = {}
        if cfg['log_weight_histograms']:
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
        
        wandb_run.log({
            'validation loss': val_score,
            **histograms
        }, commit=False)

    model.train()
    return val_score, iou_accum
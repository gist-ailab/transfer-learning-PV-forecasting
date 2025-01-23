import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import argparse

plt.switch_backend('agg')

class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values.split(","):
             k,v = kv.split("=")
             my_dict[k] = v
         setattr(namespace, self.dest, my_dict)


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        self.save_latest_checkpoint(val_loss, model, path)
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

    def save_latest_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path + '/' + 'model_latest.pth')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def visual_original(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_out(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    # Sample data
    gt_values = true        # Replace with your 'gt' data starting right after 'x'
    pd_values = preds       # Replace with your 'pd' data starting right after 'gt'

    # Create the figure and axis
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)

    # Plot the data
    total_length = len(gt_values)
    # x_positions = range(1, len(x_values) + 1)
    # gt_positions = range(len(x_values) + 1, len(x_values) + len(gt_values) + 1)
    # pd_positions = range(len(x_values) + 1, len(x_values) + len(pd_values) + 1)
    # x_positions = range(0, len(x_values))
    gt_positions = range(len(gt_values))
    pd_positions = range(len(pd_values))

    ax.plot(gt_positions, gt_values, 'cs', label='ground truth')  # gt is plotted right after x ends
    ax.plot(pd_positions, pd_values, 'rx', label='prediction')    # pd is plotted right after gt ends

    # # Adjust the x-axis to show sequential numbers
    # tick_interval = max(0, total_length // 10)  # Adjust this value as needed to reduce overlap
    # ax.set_xticks(range(0, total_length, tick_interval))
    # ax.set_xticklabels(range(0, total_length, tick_interval))

    # Additional customization (optional)
    plt.title('Prediction of PV power Generation')
    plt.xlabel(f'Time (hours), prediction length: {len(gt_values)} hours')
    plt.ylabel('Active Power[kW]')
    plt.grid(True)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    
    
    # """
    # Results visualization (GT and Pred only). temporary added for reporting @240327
    # """
    # # Sample data
    # gt_values = true        # Replace with your 'gt' data starting right after 'x'
    # pd_values = preds       # Replace with your 'pd' data starting right after 'gt'

    # # Create the figure and axis
    # plt.figure(figsize=(20, 5))
    # ax = plt.subplot(111)

    # # Plot the data
    # total_length = len(gt_values)
    # # x_positions = range(1, len(x_values) + 1)
    # # gt_positions = range(len(x_values) + 1, len(x_values) + len(gt_values) + 1)
    # # pd_positions = range(len(x_values) + 1, len(x_values) + len(pd_values) + 1)
    # gt_positions = range(len(gt_values))
    # pd_positions = range(len(pd_values))

    # ax.plot(gt_positions, gt_values, label='ground truth')  # gt is plotted right after x ends
    # ax.plot(pd_positions, pd_values, label='prediction')    # pd is plotted right after gt ends

    # # Adjust the x-axis to show sequential numbers
    # tick_interval = max(0, total_length // 10)  # Adjust this value as needed to reduce overlap
    # ax.set_xticks(range(0, total_length, tick_interval))
    # ax.set_xticklabels(range(0, total_length, tick_interval))

    # # Additional customization (optional)
    # plt.title('Prediction of PV power Generation')
    # plt.xlabel(f'Time (hours), prediction length: {len(gt_values)} hours')
    # plt.ylabel('Active Power[kW]')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(name, bbox_inches='tight')
    
    
def visual(input_seq, true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    # Sample data
    x_values = input_seq    # Replace with your 'x' data
    gt_values = true        # Replace with your 'gt' data starting right after 'x'
    pd_values = preds       # Replace with your 'pd' data starting right after 'gt'

    # Create the figure and axis
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)

    # Plot the data
    total_length = len(x_values) + len(gt_values)
    # x_positions = range(1, len(x_values) + 1)
    # gt_positions = range(len(x_values) + 1, len(x_values) + len(gt_values) + 1)
    # pd_positions = range(len(x_values) + 1, len(x_values) + len(pd_values) + 1)
    x_positions = range(0, len(x_values))
    gt_positions = range(len(x_values), len(x_values) + len(gt_values))
    pd_positions = range(len(x_values), len(x_values) + len(pd_values))

    ax.plot(x_positions, x_values, label='input data')      # x is plotted at its position
    ax.plot(gt_positions, gt_values, 'cs', label='ground truth')  # gt is plotted right after x ends
    ax.plot(pd_positions, pd_values, 'rx', label='prediction')    # pd is plotted right after gt ends

    # # Adjust the x-axis to show sequential numbers
    # tick_interval = max(0, total_length // 10)  # Adjust this value as needed to reduce overlap
    # ax.set_xticks(range(0, total_length, tick_interval))
    # ax.set_xticklabels(range(0, total_length, tick_interval))

    # Additional customization (optional)
    plt.title('Prediction of PV power Generation')
    plt.xlabel(f'Time (hours), prediction length: {len(gt_values)} hours')
    plt.ylabel('Active Power[kW]')
    plt.grid(True)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
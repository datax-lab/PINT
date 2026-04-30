import argparse

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("--cancer_type")
parser.add_argument("--exp_num")
parser.add_argument("--gpu_num")

import ast

args = parser.parse_args()
print(args)

DATASET = str(args.cancer_type)
EXP_NUM = int(args.exp_num)


DATASET_PATH = f'../exp_setup/data_splits/{EXP_NUM}'
PATHWAY_PATH = f'../processed_data'

gpu_id = int(args.gpu_num)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import math
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler, BatchSampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import ray
from ray import tune
from ray import train
from ray.air import session
import ray.cloudpickle as pickle
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train import Checkpoint, get_checkpoint

from sklearn.manifold import TSNE


class OmicsDataset(Dataset):
    
    def __init__(self, rna, time, event):
        
        self.dtype = torch.FloatTensor
        self.rna   = torch.from_numpy(rna).type(self.dtype)
        self.time  = torch.from_numpy(time).type(self.dtype)
        self.event = torch.from_numpy(event).type(self.dtype)

    def __getitem__(self, index):
        return self.rna[index], self.time[index], self.event[index]

    def __len__(self):
        return len(self.rna)


def sort_collate_fn(batch):
    
    batch.sort(key=lambda x: x[1], reverse=True)

    rna, time, event = zip(*batch)
    return (
        torch.stack(rna),
        torch.stack(time),
        torch.stack(event),
    )


def sort_data(path):
    ''' sort the genomic and clinical data w.r.t. survival time (OS_MONTHS) in descending order
    Input:
        path: path to input dataset (which is expected to be a csv file).
    Output:
        x: sorted genomic inputs.
        ytime: sorted survival time (OS_MONTHS) corresponding to 'x'.
        yevent: sorted censoring status (OS_EVENT) corresponding to 'x', where 1 --> deceased; 0 --> censored.
        age: sorted age corresponding to 'x'.
    '''
    
    data = pd.read_csv(path)

    data.sort_values("OS_MONTHS", ascending=False, inplace=True)

    x = data.drop(["OS_MONTHS", "OS_STATUS"], axis=1).values
    ytime = data.loc[:, ["OS_MONTHS"]].values
    yevent = data.loc[:, ["OS_STATUS"]].values
    
    return (x, ytime, yevent)


def R_set(x):
    '''Create an indicator matrix of risk sets, where T_j >= T_i.
    Note that the input data have been sorted in descending order.
    Input:
        x: a PyTorch tensor that the number of rows is equal to the number of samples.
    Output:
        indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
    '''
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return(indicator_matrix)


def attention_entropy(alpha, eps=1e-12):
    """
    alpha: (B, P) attention weights
    returns: scalar normalized entropy
    """
    P = alpha.size(1)
    entropy = -(alpha * (alpha + eps).log()).sum(dim=1)   # (B,)
    entropy = entropy / math.log(P)                       # normalize
    return entropy.mean()


def neg_par_log_likelihood(pred, ytime, yevent):
    '''Calculate the average Cox negative partial log-likelihood.
    Input:
        pred: linear predictors from trained model.
        ytime: true survival time from load_data().
        yevent: true censoring status from load_data().
    Output:
        cost: the cost that is to be minimized.
    '''
    n_observed = yevent.sum(0)
    ytime_indicator = R_set(ytime)
    ###if gpu is being used
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
    ###
    risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent.view(-1, 1))
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    return cost


def c_index(pred, ytime, yevent):
    '''Calculate concordance index to evaluate models.
    Input:
        pred: linear predictors from trained model.
        ytime: true survival time from load_data().
        yevent: true censoring status from load_data().
    Output:
        Concordance_index: c-index (between 0 and 1).
    '''
    n_sample = len(ytime)
    ytime_indicator = R_set(ytime)
    ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
    ###T_i is uncensored
    censor_idx = (yevent == 0).nonzero()
    zeros = torch.zeros(n_sample)
    ytime_matrix[censor_idx, :] = zeros
    ###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
    pred_matrix = torch.zeros_like(ytime_matrix)
    for j in range(n_sample):
        for i in range(n_sample):
            if pred[i] < pred[j]:
                pred_matrix[j, i]  = 1
            elif pred[i] == pred[j]: 
                pred_matrix[j, i] = 0.5

    concord_matrix = pred_matrix.mul(ytime_matrix)
    ###numerator
    concord = torch.sum(concord_matrix)
    ###denominator
    epsilon = torch.sum(ytime_matrix)
    ###c-index = numerator/denominator
    concordance_index = torch.div(concord, epsilon)
    ###if gpu is being used
    if torch.cuda.is_available():
        concordance_index = concordance_index.cuda()
    ###
    return(concordance_index)
    
    
class PathwayEncoder(nn.Module):
    
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 embed_dim,
                 activation_fn, 
                 dropout1,
                 dropout2):
        
        super().__init__()
                
        act_dict = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'sigmoid': nn.Sigmoid}

        self.pe_linear_layer1 = nn.Linear(input_dim, hidden_dim)
        self.pe_act_layer1 = act_dict.get(activation_fn, nn.ReLU)()
        self.pe_dropout1 = nn.Dropout(dropout1)
        
        self.pe_linear_layer2 = nn.Linear(hidden_dim, embed_dim)
        self.pe_act_layer2 = act_dict.get(activation_fn, nn.ReLU)()
        self.pe_dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        x = self.pe_linear_layer1(x)
        x = self.pe_act_layer1(x)
        x = self.pe_dropout1(x)
        
        x = self.pe_linear_layer2(x)
        x = self.pe_dropout2(x)
        #x = self.pe_act_layer2(x)
        
        return x
    
    
class MHSA(nn.Module):
    
    def __init__(self, embed_dim, num_heads, attn_dropout):
        
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.D = embed_dim
        self.H = num_heads
        self.dk = embed_dim // num_heads
        
        self.q_proj = nn.Linear(self.D, self.D, bias=False)
        
        self.k_proj = nn.Linear(self.D, self.D, bias=False)
        
        self.v_proj = nn.Linear(self.D, self.D, bias=False)
        
        self.out_proj = nn.Linear(self.D, self.D, bias=False)
        
        self.dropout = nn.Dropout(attn_dropout)
        
        self.ln = nn.LayerNorm(self.D)

        
    def forward(self, X):
        
        # Get the number of samples, number of pathways, and the total number of genes
        B, P, D = X.shape
        
        # Pass the input through LayerNorm
        h = self.ln(X)
        
        # Pass 'h' through Query, Key and Value layers...
        Q, K, V = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        
        def split(t): return t.view(B, P, self.H, self.dk).transpose(1, 2)
        
        Q, K, V = map(split, (Q, K, V))
        
        scores = (Q @ K.transpose(-2, -1)) / (self.dk ** 0.5)  # (B,H,P,P)
        
        attn = torch.softmax(scores, dim=-1)
        
        attn = self.dropout(attn)
        
        context = attn @ V
        context = context.transpose(1, 2).reshape(B, P, D)
        
        out = self.out_proj(context)
        
        return X + out, attn
    

class AttentivePooling(nn.Module):
    
    def __init__(self, embed_dim):
        super().__init__()
        self.W = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, 1, bias=False)
        
        
    def forward(self, Z):
        
        H = torch.tanh(self.W(Z))
        
        scores = self.v(H).squeeze(-1)
        # print(f'scores.shape: {scores.shape}')
        
        alpha = torch.softmax(scores, dim=-1)
        # print(f'alpha.shape: {alpha.shape}')
        
        S = torch.einsum("bp,bpd->bd", alpha, Z)
        # print(f'S.shape: {S.shape}')
        
        return S, alpha
    
    
class FFN(nn.Module):
    
    def __init__(self, embed_dim, ffn_mult, ffn_dropout):
        
        super().__init__()
        
        self.ln = nn.LayerNorm(embed_dim)  
        
        self.fc1 = nn.Linear(embed_dim, ffn_mult * embed_dim)
        self.fc2 = nn.Linear(ffn_mult * embed_dim, embed_dim)
        self.dropout = nn.Dropout(ffn_dropout)


    def forward(self, X):
        
        h = self.ln(X)
        
        h = F.gelu(self.fc1(h))
        
        h = self.dropout(h)
        
        h = self.fc2(h)
        
        h = self.dropout(h)
        
        return X + h


class PathwayEncodersWithSelfAttn(nn.Module):
    
    def __init__(self, 
                 pathway_mask,
                 hidden_dim,
                 embed_dim,
                 activation_fn,
                 dropout1,
                 dropout2,
                 num_heads,
                 attn_dropout,
                 num_attn_layers,
                 ffn_mult,
                 ffn_dropout):
        
        super().__init__()
        
        # Check if the pathway_mask is 2D
        assert pathway_mask.dim() == 2
        
        # Retrieve the number of pathways and the total number of genes
        self.P, self.G = pathway_mask.shape
        
        # Each pathway will be embedded as a vector of embed_dim
        self.embed_dim = embed_dim
        
        pmask = pathway_mask.to(torch.bool)
        self.register_buffer("pathway_mask_buf", pmask, persistent=False)

        self.pathway_indices = []
        
        encoders = []
        
        # Iterate through each pathway
        for p in range(self.P):
            
            idx = torch.nonzero(self.pathway_mask_buf[p], as_tuple=False).flatten()
            
            self.pathway_indices.append(idx)
            
            encoders.append(PathwayEncoder(idx.numel(), hidden_dim, embed_dim, activation_fn, dropout1, dropout2))
        
        self.encoders = nn.ModuleList(encoders)

        blocks = []
        
        for _ in range(num_attn_layers):
            
            blocks += [MHSA(embed_dim, num_heads=num_heads, attn_dropout=attn_dropout),
                       FFN(embed_dim, ffn_mult=ffn_mult, ffn_dropout=ffn_dropout)]
        
        self.attn_blocks = nn.ModuleList(blocks)
        
        self.pool = AttentivePooling(embed_dim)
        

    def encode_pathways(self, X):        
        
        Zs = []
        
        # Iterate through each pathway
        for p in range(self.P):
            
            # Retrieve the corresponding active gene indices in that pathway
            idx = self.pathway_indices[p]
            
            # Slice the input to select data from the corresponding cells in the pathway
            x_p = X[:, p, idx.to(X.device)]
            
            # Forward propagate through the encoders and append the output into Zs
            Zs.append(self.encoders[p](x_p))
        
        # Stack the output and return them
        return torch.stack(Zs, dim=1)
    

    def forward(self, X):
        
        # Ensure that we are receiving the input correctly...
        assert X.dim() == 3 and X.shape[1] == self.P and X.shape[2] == self.G
        
        # Get the batch-size
        B = X.size(0)

        Z = self.encode_pathways(X)
        
        attn_last = None
        
        for layer in self.attn_blocks:
            
            if isinstance(layer, MHSA):
                H, attn_last = layer(Z)
            
            else:
                H = layer(H)
        
        Z_revised = H
        
        S, alpha_pool = self.pool(Z_revised)
        
        return Z_revised, S, attn_last, alpha_pool
    
    
class ProjectionHead(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim))
        
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
    

class ClassifierHead(torch.nn.Module):
    
    def __init__(self, in_dim, hidden_dim, clf_activation_fn, clf_dropout, num_classes):
        
        super().__init__()
        
        act_dict = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'sigmoid': nn.Sigmoid}
        
        self.clf_linear        = torch.nn.Linear(in_dim, hidden_dim)
        self.clf_activation_fn = act_dict.get(clf_activation_fn, nn.ReLU)()
        self.clf_dropout       = torch.nn.Dropout(clf_dropout)        
        # self.clf_final_layer = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.clf_final_layer   = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.clf_final_layer.weight.data.uniform_(-0.001, 0.001)

    def forward(self, x): 
        x = self.clf_linear(x)
        x = self.clf_activation_fn(x)
        x = self.clf_dropout(x)
        x = self.clf_final_layer(x)
        return x


class LabeledPathwayDataset(Dataset):
    """X: (N,P,G), y: (N,)"""
    
    def __init__(self, X, y):
        assert X.dim() == 3 and y.dim() == 1 and X.size(0) == y.size(0)
        self.X, self.y = X, y.long()
    
    def __len__(self): 
        return self.X.size(0)
    
    def __getitem__(self, i): 
        return self.X[i], self.y[i]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def get_lrs(opt):
    return [pg["lr"] for pg in opt.param_groups]


def set_lrs(opt, lrs):
    for pg, lr in zip(opt.param_groups, lrs):
        pg["lr"] = lr


def fit_model(config, trial_id):    
    enc_hidden_dim = config['dims'][0]
    enc_embed_dim  = config['dims'][1]
    enc_activation_fn = config['enc_activation_fn']
    enc_dropout1 = config['enc_dropout1']
    enc_dropout2 = config['enc_dropout2']
    
    attn_num_heads = config['attn_num_heads']
    attn_dropout   = config['attn_dropout']
    
    num_attn_layers = config['num_attn_layers']
    ffn_mult        = config['ffn_mult']
    ffn_dropout     = config['ffn_dropout']
    
    clf_hidden_dim    = config['dims'][2]
    clf_activation_fn = config['clf_activation_fn'] 
    clf_dropout       = config['clf_dropout']
    
    learning_rate = config['learning_rate']
    weight_decay  = config['weight_decay']

    num_epochs = config['num_epochs']
    batch_size = config['batch_size']

    warmup_start_factor = config['warmup_start_factor']
    warmup_epochs       = config['warmup_epochs']
    
    cosine_eta_min = config['cosine_eta_min']
    
    # lr_factor   = config['lr_factor']
    # lr_patience = config['lr_patience']
    
    lam_entropy = config['lam_entropy'] #1e-3  # safe default
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = np.load(f'{DATASET_PATH}/X_train_3d_{EXP_NUM}.npy'), np.load(f'{DATASET_PATH}/y_train_{EXP_NUM}.npy')
    y_train_time     = y_train[:, 0]
    y_train_event    = y_train[:, 1]
    
    X_val, y_val = np.load(f'{DATASET_PATH}/X_val_3d_{EXP_NUM}.npy'), np.load(f'{DATASET_PATH}/y_val_{EXP_NUM}.npy')
    y_val_time   = y_val[:, 0]
    y_val_event  = y_val[:, 1]
    
    X_test, y_test = np.load(f'{DATASET_PATH}/X_test_3d_{EXP_NUM}.npy'), np.load(f'{DATASET_PATH}/y_test_{EXP_NUM}.npy')
    y_test_time    = y_test[:, 0]
    y_test_event   = y_test[:, 1]
    
    pathway_mask = np.load(f'{PATHWAY_PATH}/pathway_mask.npy')
    
    dtype = torch.FloatTensor
    pathway_mask = torch.from_numpy(pathway_mask).type(dtype)
    
    train_dataset = OmicsDataset(X_train, y_train_time, y_train_event)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sort_collate_fn)
    
    val_dataset   = OmicsDataset(X_val, y_val_time, y_val_event)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=sort_collate_fn)
    
    test_dataset   = OmicsDataset(X_test, y_test_time, y_test_event)
    test_loader    = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=sort_collate_fn)

    model = PathwayEncodersWithSelfAttn(pathway_mask=pathway_mask,
                                        hidden_dim = enc_hidden_dim,
                                        embed_dim = enc_embed_dim, 
                                        activation_fn = enc_activation_fn,
                                        dropout1 = enc_dropout1,
                                        dropout2 = enc_dropout2,
                                        num_heads = attn_num_heads,
                                        attn_dropout = attn_dropout,
                                        num_attn_layers = num_attn_layers, 
                                        ffn_mult = ffn_mult,
                                        ffn_dropout = ffn_dropout
                                       )

    clf = ClassifierHead(enc_embed_dim, 
                         clf_hidden_dim, 
                         clf_activation_fn, 
                         clf_dropout, 
                         num_classes=1).to(device)

    params = list(model.parameters()) + list(clf.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    warmup_epochs  = max(1, int(warmup_epochs * num_epochs))

    # Step 1: Linear warmup for first 5% of epochs
    warmup = LinearLR(optimizer, start_factor=warmup_start_factor, total_iters=warmup_epochs)

    # Step 2: Cosine decay for the rest
    cosine = CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs), eta_min=cosine_eta_min)

    # Combine both sequentially
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    
    # plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                      mode="min",
    #                                                      factor=lr_factor,
    #                                                      patience=lr_patience,
    #                                                      threshold=1e-4,
    #                                                      cooldown=5,
    #                                                      min_lr=cosine_eta_min)
    model.to(device) 
    clf.to(device)    
    
    train_total_loss_values = []
    val_total_loss_values = []

    train_c_indices = []
    val_c_indices = []
    
    lr_values = []
    
    best_val_cindex   = -1
    best_train_cindex = -1
    best_train_loss   = np.inf
    best_val_loss     = np.inf
    
    tol_count = 0
    
    for ep in range(1, num_epochs + 1):
        
        if ep >= 200 and tol_count >= 50:
            
            epochs = range(1, len(train_total_loss_values) + 1)

            metrics = [
                ("Total Loss", train_total_loss_values, val_total_loss_values),
                ("C-Index", train_c_indices, val_c_indices),
                ("Learning Rate", lr_values, None)
            ]

            # Create 4×2 grid of subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            axes = axes.ravel()

            for i, (title, train_vals, val_vals) in enumerate(metrics):
                ax = axes[i]
                ax.plot(epochs, train_vals, label="Train", color="tab:blue", linewidth=2)
                if val_vals is not None:
                    ax.plot(epochs, val_vals, label="Validation", color="tab:orange", linewidth=2)
                ax.set_title(title, fontsize=13, fontweight='bold')
                ax.set_xlabel("Epochs")
                ax.set_ylabel(title)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend()

            # Adjust layout for better spacing
            plt.tight_layout()
            plt.savefig(f"plots_new/{EXP_NUM}/{trial_id}_plts.png")
            plt.clf()
 
            break
        
        model.train()
        clf.train()
        
        train_total_loss = 0.0
        train_num_batches = 0        
        train_preds = []
        train_times = []
        train_events = []
    
        for idx, (train_rna, train_time, train_event) in enumerate(train_loader):
            
            if torch.cuda.is_available():
                train_rna, train_time, train_event = train_rna.cuda(), train_time.cuda(), train_event.cuda()
            
            # Zero out the existing gradients
            optimizer.zero_grad(set_to_none=True)
            
            _, S, _, alpha_pool = model(train_rna)
            train_pred = clf(S)
            
            if train_event.sum(0)==0:
                continue

            cox_loss = neg_par_log_likelihood(train_pred, train_time, train_event)
            entropy_loss = attention_entropy(alpha_pool)    
            train_epoch_loss = cox_loss + lam_entropy * entropy_loss

            train_epoch_loss.backward()
            optimizer.step()

            train_total_loss  += train_epoch_loss.item()
            train_num_batches += 1    
            train_preds.append(train_pred.detach().cpu())
            train_times.append(train_time.detach().cpu().view(-1))
            train_events.append(train_event.detach().cpu().view(-1))
                
        train_preds  = torch.cat(train_preds,  dim=0)  # (N,)
        train_times  = torch.cat(train_times,  dim=0)  # (N,)
        train_events = torch.cat(train_events, dim=0)  # (N,)
        
        train_total_loss /= train_num_batches
        
        with torch.no_grad():
            model.eval() 
            clf.eval()

            val_total_loss = 0.0
            val_num_batches = 0        
            val_preds = []
            val_times = []
            val_events = []
        
            for idx, (val_rna, val_time, val_event) in enumerate(val_loader):
                if torch.cuda.is_available():
                    val_rna, val_time, val_event = val_rna.cuda(), val_time.cuda(), val_event.cuda()

                _, S, _, alpha_pool = model(val_rna)
                val_pred = clf(S)
            
                cox_loss = neg_par_log_likelihood(val_pred, val_time, val_event)
                entropy_loss = attention_entropy(alpha_pool)    
                val_epoch_loss = cox_loss + lam_entropy * entropy_loss            
                val_total_loss  += val_epoch_loss.item()
                val_num_batches += 1
            
                val_preds.append(val_pred.detach().cpu())
                val_times.append(val_time.detach().cpu().view(-1))
                val_events.append(val_event.detach().cpu().view(-1))

            val_preds  = torch.cat(val_preds,  dim=0)  # (N,)
            val_times  = torch.cat(val_times,  dim=0)  # (N,)
            val_events = torch.cat(val_events, dim=0)  # (N,)

            val_total_loss /= val_num_batches
            

        train_total_loss_values.append(train_total_loss)
        val_total_loss_values.append(val_total_loss)
        
        scheduler.step()
        
        # Cache LRs *before* plateau
        # lrs_before = get_lrs(optimizer)
    
        # plateau.step(val_total_loss)
        
        # If Plateau reduced LR, sync cosine.base_lrs so the drop persists
#         lrs_after = get_lrs(optimizer)
        
#         plateau_fired = any(a < b - 1e-12 for a, b in zip(lrs_after, lrs_before))
#         if plateau_fired:
#             # IMPORTANT: update cosine's base_lrs so future cosine values are lower
#             cosine.base_lrs = lrs_after[:]  # copy current (reduced) LRs into cosine baseline

        lr_values.append(optimizer.param_groups[0]["lr"])
        
        train_times, indices = torch.sort(train_times, descending=True)
        train_events         = train_events[indices]
        train_preds          = train_preds[indices]
        train_cindex         = c_index(train_preds, train_times, train_events)
        train_c_indices.append(train_cindex.item())

        val_times, indices = torch.sort(val_times, descending=True)
        val_events         = val_events[indices]
        val_preds          = val_preds[indices]
        val_cindex         = c_index(val_preds, val_times, val_events)
        val_c_indices.append(val_cindex.item())

        print(f'ep: {ep}, train_loss: {train_total_loss}, val_loss: {val_total_loss}, train_c_index: {train_cindex.item()}, val_c_index: {val_cindex.item()}')
        
        if val_cindex.item() > best_val_cindex:
            tol_count = 0
            best_val_c_index_epoch = ep
            
            best_train_loss     = train_total_loss
            best_val_loss       = val_total_loss
            
            best_train_cindex  = train_cindex.item()
            best_val_cindex    = val_cindex.item()
            
            torch.save(model.state_dict(), f"chkpoints_new/{EXP_NUM}/{trial_id}_model.pth")
            torch.save(clf.state_dict(), f"chkpoints_new/{EXP_NUM}/{trial_id}_clf.pth")

        else:
            tol_count += 1
            print(f'tol_count: {tol_count}')
            
        if ep % 50 == 0:
            epochs = range(1, len(train_total_loss_values) + 1)

            metrics = [
                ("Total Loss", train_total_loss_values, val_total_loss_values),
                ("C-Index", train_c_indices, val_c_indices),
                ("Learning Rate", lr_values, None)
            ]

            # Create 4×2 grid of subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            axes = axes.ravel()

            for i, (title, train_vals, val_vals) in enumerate(metrics):
                ax = axes[i]
                ax.plot(epochs, train_vals, label="Train", color="tab:blue", linewidth=2)
                if val_vals is not None:
                    ax.plot(epochs, val_vals, label="Validation", color="tab:orange", linewidth=2)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel("Epochs")
                ax.set_ylabel(title)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend()

            # Adjust layout for better spacing
            plt.tight_layout()
            plt.savefig(f"plots_new/{EXP_NUM}/{trial_id}_plts.png")
            plt.clf()
    
    model.load_state_dict(torch.load(f"chkpoints_new/{EXP_NUM}/{trial_id}_model.pth"))
    clf.load_state_dict(torch.load(f"chkpoints_new/{EXP_NUM}/{trial_id}_clf.pth"))
    
    with torch.no_grad():
        model.eval() 
        clf.eval()

        test_preds  = []
        test_times  = []
        test_events = []

        for idx, (test_rna, test_time, test_event) in enumerate(test_loader):

            if torch.cuda.is_available():
                test_rna, test_time, test_event = test_rna.cuda(), test_time.cuda(), test_event.cuda()

            _, S, _, alpha_pool = model(test_rna)
            test_pred = clf(S)

            test_preds.append(test_pred.detach().cpu())
            test_times.append(test_time.detach().cpu().view(-1))
            test_events.append(test_event.detach().cpu().view(-1))

        test_preds  = torch.cat(test_preds,  dim=0)
        test_times  = torch.cat(test_times,  dim=0)
        test_events = torch.cat(test_events, dim=0)

        test_times, indices = torch.sort(test_times, descending=True)
        test_events         = test_events[indices]
        test_preds          = test_preds[indices]
        test_cindex         = c_index(test_preds, test_times, test_events)
    
    return {
        'trial_id': trial_id,
        'best_val_c_index_epoch': best_val_c_index_epoch,
        'best_train_loss':        best_train_loss,
        'best_val_loss':          best_val_loss,
        'best_train_cindex':      best_train_cindex,
        'best_val_cindex':        best_val_cindex,
        'test_c_index':           test_cindex.item()
    }


if __name__ == "__main__":

    config = {}
    config["enc_hidden_dim"] = 32
    config["enc_embed_dim"]  = 32
    config["enc_activation_fn"] = 'gelu'
    config["enc_dropout1"] = 0.21187363026627393
    config["enc_dropout2"] = 0.06001151009679313
    
    config["attn_num_heads"] = 2
    config["attn_dropout"]   = 0.03671569700655142
    
    config["num_attn_layers"] = 1
    config["ffn_mult"]        = 1
    config["ffn_dropout"]     = 0.0642248051433565
    
    config["clf_hidden_dim"]    = 32
    config["clf_activation_fn"] = 'tanh'
    config["clf_dropout"]       = 0.11784858566467032
    
    config["learning_rate"] = 5e-05
    config["weight_decay"]  = 0.001

    config["num_epochs"] = 600
    config["batch_size"] = 256

    config["warmup_start_factor"] = 0.01
    config["warmup_epochs"]       = 0.05
    
    config["cosine_eta_min"] = 1e-07
    
    config["lam_entropy"] = 0.0003
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = fit_model(config, trial_id)
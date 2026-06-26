from __future__ import print_function
import argparse
from timeit import default_timer as timer
from argparse import Namespace
import numpy as np
from dataset_survival_lung import Generic_MIL_Survival_Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import LayerNorm, ReLU
from torch.utils.data import DataLoader,WeightedRandomSampler,RandomSampler,SequentialSampler,Sampler
from torch.utils.data.dataloader import default_collate
import os
import torch_geometric.utils
import torch_geometric.data
import torch_geometric.sampler
import torch_geometric.loader
import torch_geometric.transforms
import torch_geometric.datasets
import torch_geometric.nn
import torch_geometric.explain
import torch_geometric.profile
from torch_geometric.nn import GATConv, DeepGCNLayer
from einops import rearrange

def get_custom_exp_code(args):
    ### New exp_code
    exp_code = '_'.join(args.split_dir.split('_')[:2])
    dataset_path = 'dataset_csv'
    param_code = ''

    if args.model_type == 'patchgatv1':
        param_code += 'PatchGATv1'
    else:
        raise NotImplementedError

    if args.resample > 0:
        param_code += '_resample'

    param_code += '_%s' % args.bag_loss

    param_code += '_a%s' % str(args.alpha_surv)

    if args.lr != 2e-4:
        param_code += '_lr%s' % format(args.lr, '.0e')

    if args.reg_type != 'None':
        param_code += '_reg%s' % format(args.lambda_reg, '.0e')

    param_code += '_%s' % args.which_splits.split("_")[0]

    if args.batch_size != 1:
        param_code += '_b%s' % str(args.batch_size)

    if args.gc != 1:
        param_code += '_gc%s' % str(args.gc)

    args.exp_code = exp_code + '_' + param_code
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def collate_MIL_survival_graph(batch):
    elem = batch[0]
    elem_type = type(elem)
    transposed = zip(*batch)
    return [samples[0] if isinstance(samples[0], torch_geometric.data.Batch) else default_collate(samples) for samples in transposed]

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

def get_split_loader(split_dataset, training=False, testing=False, weighted=False, mode='coattn', batch_size=1):
    collate = collate_MIL_survival_graph

    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size,
                                    sampler=WeightedRandomSampler(weights, len(weights)), collate_fn=collate, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler=RandomSampler(split_dataset),
                                    collate_fn=collate, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler=SequentialSampler(split_dataset),
                                collate_fn=collate, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset) * 0.1)), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids), collate_fn=collate,
                            **kwargs)

    return loader

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

def init_max_weights(module):
    import math
    import torch.nn as nn

    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

class Embedding(nn.Module):  # Patch Embedding + Position Embedding + Class Embedding
    def __init__(self, image_channels=3, image_size=224, patch_size=16, dim=768, drop_ratio=0.):
        super(Embedding, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2  # Patch数量

        self.patch_conv = nn.Conv2d(image_channels, dim, patch_size, patch_size)  # 使用卷积将图像划分成Patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # class embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))  # position embedding
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.patch_conv(x)
        x = rearrange(x, "B C H W -> B (H W) C")
        cls_token = torch.repeat_interleave(self.cls_token, x.shape[0], dim=0)  # (1,1,dim) -> (B,1,dim)
        x = torch.cat([cls_token, x], dim=1)  # (B,1,dim) cat (B,num_patches,dim) --> (B,num_patches+1,dim)
        x = x + self.pos_emb
        return self.dropout(x)  # token

class MultiHeadAttention(nn.Module):  # Multi-Head Attention
    def __init__(self, dim, num_heads=8, drop_ratio=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)  # 使用一个Linear，计算得到qkv
        self.dropout = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # B: Batch Size / P: Num of Patches / D: Dim of Patch / H: Num of Heads / d: Dim of Head
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "B P (C H d) -> C B H P d", C=3, H=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离qkv
        k = rearrange(k, "B H P d -> B H d P")
        # Attention(Q, K, V ) = softmax(QKT/dk)V （T表示转置)
        attn = torch.matmul(q, k) * self.head_dim ** -0.5  # QKT/dk
        attn = F.softmax(attn, dim=-1)  # softmax(QKT/dk)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)  # softmax(QKT/dk)V
        x = rearrange(x, "B H P d -> B P (H d)")
        x = self.proj(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):  # MLP
    def __init__(self, in_dims, hidden_dims=None, drop_ratio=0.):
        super(MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = in_dims * 4  # linear的hidden_dims默认为in_dims的4倍

        self.fc1 = nn.Linear(in_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, in_dims)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        # Linear + GELU + Dropout + Linear + Dropout
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):  # Transformer Encoder Block
    def __init__(self, dim, num_heads=8, drop_ratio=0.):
        super(EncoderBlock, self).__init__()

        self.layernorm1 = nn.LayerNorm(dim)
        self.multiheadattn = MultiHeadAttention(dim, num_heads)
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        # 两次残差连接，分别在Multi-Head Attention和MLP之后
        x0 = x
        x = self.layernorm1(x)
        x = self.multiheadattn(x)
        x = self.dropout(x)
        x1 = x + x0  # 第一次残差连接
        x = self.layernorm2(x1)
        x = self.mlp(x)
        x = self.dropout(x)
        return x + x1  # 第二次残差连接

class MLPHead(nn.Module):  # MLP Head
    def __init__(self, dim, num_classes=1000):
        super(MLPHead, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        # 对于一般数据集，此处为1层Linear; 对于ImageNet-21k数据集，此处为Linear+Tanh+Linear
        self.mlphead = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.layernorm(x)
        cls = x[:, 0, :]  # 去除class token
        return self.mlphead(cls)

class ViT(nn.Module):  # Vision Transformer
    def __init__(self, image_channels=3, image_size=224, num_classes=1000, patch_size=16, dim=768, num_heads=12,
                 layers=12):
        super(ViT, self).__init__()
        self.embedding = Embedding(image_channels, image_size, patch_size, dim)
        self.encoder = nn.Sequential(
            *[EncoderBlock(dim, num_heads) for i in range(layers)]  # encoder结构为layers(L)个Transformer Encoder Block
        )
        self.head = MLPHead(dim, num_classes)

    def forward(self, x):
        x_emb = self.embedding(x)
        feature = self.encoder(x_emb)
        return self.head(feature)

def vit_base(num_classes=1000):  # ViT-Base
    return ViT(image_channels=1, image_size=150, num_classes=num_classes, patch_size=16, dim=768, num_heads=12,
               layers=12)

class PatchGAT_Survv1(torch.nn.Module):
    def __init__(self, input_dim=2227, num_layers=4, edge_agg='spatial', multires=False, resample=0,
                 fusion=None, num_features=1024, hidden_dim=128, linear_dim=64, use_edges=False, pool=False,
                 dropout=0.25, n_classes=4):
        super(PatchGAT_Survv1, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers - 1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GATConv(hidden_dim, hidden_dim, heads=1)
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim * 4, D=hidden_dim * 4, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout)])

        self.vb = vit_base(num_classes=512)
        self.classifier = torch.nn.Linear(hidden_dim * 4 + 512, n_classes)

    def forward(self, **kwargs):
        data = kwargs['x_path']
        data1 = kwargs['x_path1']

        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent

        batch = data.batch
        edge_attr = None

        x = self.fc(data.x)
        x_ = x

        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        h1 = self.vb(data1)
        h = torch.cat((h, h1.squeeze(0)), 0)
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        return Y_hat

def summary_survival(model, loader):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()

    for batch_idx, (ctdata, data_WSI, label, event_time, c) in enumerate(loader):

        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 100_000:
                continue

        ctdata, = ctdata.type(torch.FloatTensor).to(device)
        ctdata = ctdata.reshape((1, 1, 150, 150))

        data_WSI = data_WSI.to(device)

        with torch.no_grad():
            Y_hat = model(x_path=data_WSI, x_path1=ctdata)

def validate(datasets: tuple, args: Namespace, model_file: str):
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    print('Done!')
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit Model...', end=' ')
    model_dict = {'num_layers': args.num_gcn_layers, 'edge_agg': args.edge_agg, 'resample': args.resample,
                  'n_classes': args.n_classes}
    model = PatchGAT_Survv1(**model_dict)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    print('Done!')

    print('\nInit Loaders...', end=' ')
    val_loader = get_split_loader(val_split, testing=args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    model.load_state_dict(torch.load(model_file))
    summary_survival(model, val_loader)

def main(args):
    start = timer()

    model_file = "./model_checkpoint.pt"
    train_dataset, val_dataset = dataset.return_splits(from_id=False, csv_path="./data_split.csv")

    print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
    datasets = (train_dataset, val_dataset)
    print("validate dataset")
    print(model_file)
    validate(datasets, args, model_file)
    end = timer()
    print('Time: %f seconds' % (end - start))

### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir', type=str, default="./graph_files/", help='data directory')
parser.add_argument('--seed',            type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--results_dir',     type=str, default='./results', help='Results directory (Default: ./results)')
parser.add_argument('--testing',         action='store_true', default=False, help='debugging tool')

### Model Parameters.
parser.add_argument('--mode',            type=str, default='graph', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--num_gcn_layers',  type=int, default=4, help = '# of GCN layers to use.')
parser.add_argument('--edge_agg',        type=str, default='spatial', help="What edge relationship to use for aggregation.")
parser.add_argument('--resample',        type=float, default=0.00, help='Dropping out random patches.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=100, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',              type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--reg',             type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')


args = parser.parse_args()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

### Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)
args.task = 'WSI_survival'
print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'seed': args.seed,
            'gc': args.gc,
            'opt': args.opt}
print('\nLoad Dataset')

if 'survival' in args.task:
    args.n_classes = 4
    study = '_'.join(args.task.split('_')[:2])
    combined_study = study
    study_dir = '%s_20x_features' % combined_study
    dataset = Generic_MIL_Survival_Dataset(csv_path = './data.csv' ,
                                           mode = args.mode,
                                           data_dir= "./graph_files/",
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
else:
    raise NotImplementedError

if isinstance(dataset, Generic_MIL_Survival_Dataset):
    args.task_type = 'survival'
else:
    raise NotImplementedError
    
if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))

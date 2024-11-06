import numpy as np
import torch
import random
from torch_geometric.data import HeteroData
from torch_geometric.transforms import NormalizeFeatures
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from model import Modelattn, ModelattnFuse
import tqdm
import torch.nn.functional as F
from util import validate, test
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from link_split import RandomLinkSplitNeg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
parser = argparse.ArgumentParser(description='train_nodesplit')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden channel')
parser.add_argument('--model', type=str, default='cnn', help='model type')
parser.add_argument('--edge_weight', type=bool, default=False, help='edge weight')
parser.add_argument('--edge_weight_scale', type=bool, default=False, help='edge weight')
parser.add_argument('--disjoint_rate', type=float, default=0, help='disjoint training rate')
parser.add_argument('--val_rate', type=float, default=0.1, help='validation')
args = parser.parse_args()
fix_seed=args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
np.random.seed(fix_seed)

batch_size = 10240
data = HeteroData()
edge_mp = np.load('data/edge_mp.npy')
edge_mm = np.load('data/edge_mm_lm.npy')
edge_pp = np.load('data/edge_pp_lm.npy')
edge_mpw = np.load('data/edge_mp_weight.npy')
if args.edge_weight:
    edge_mpw = np.load('data/edge_mp_weight.npy')
    edge_mmw = np.load('data/edge_mm_weight.npy')
    edge_ppw = np.load('data/edge_pp_weight.npy')
    if args.edge_weight_scale:
        epsilon = 0.01
        edge_mmw = epsilon + (1 - 2 * epsilon) * ((edge_mmw - edge_mmw.min()) / (edge_mmw.max()-edge_mmw.min()))
        edge_ppw = epsilon + (1 - 2 * epsilon) * ((edge_ppw - edge_ppw.min()) / (edge_ppw.max()-edge_ppw.min()))
if args.model == 'cnn':
    data['mhc'].x = torch.Tensor(np.load('data/mhc_2d_lm.npy')).permute(0, 2, 1)
    data['pt'].x = torch.Tensor(np.load('data/pt_2d_lm.npy')).permute(0, 2, 1)
n = 178099 # 178099
data['mhc', 'bind', 'pt'].edge_index = torch.Tensor(edge_mp)[:,:-n].to(torch.long)
data['mhc', 'mm', 'mhc'].edge_index = torch.Tensor(edge_mm).to(torch.long)
data['pt', 'pp', 'pt'].edge_index = torch.Tensor(edge_pp).to(torch.long)
edge_neg = torch.Tensor(edge_mp)[:,-n:].to(torch.long)
data['mhc', 'bind', 'pt'].edge_attr = torch.Tensor(edge_mpw)
if args.edge_weight:
    data['mhc', 'bind', 'pt'].edge_attr = torch.Tensor(edge_mpw)
    data['mhc', 'mm', 'mhc'].edge_attr = torch.Tensor(edge_mmw)
    data['pt', 'pp', 'pt'].edge_attr = torch.Tensor(edge_ppw)

transform = NormalizeFeatures()
data = transform(data)
scale = int(edge_neg.shape[1] / (data['mhc', 'bind', 'pt'].edge_index.shape[1])) 
data = T.ToUndirected()(data)
transform = RandomLinkSplitNeg(edge_neg, disjoint_train_ratio=args.disjoint_rate, num_val=args.val_rate, num_test=0.0, edge_types=['bind'], add_negative_train_samples=False, )
train_data, val_data, test_data = transform(data)
edge_label_index = train_data['mhc', 'bind', 'pt'].edge_label_index
edge_label = train_data['mhc', 'bind', 'pt'].edge_label
edge_val = train_data['mhc', 'bind', 'pt'].edge_attr
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 20],
    # neg_sampling_ratio=1.0,
    edge_label_index=(('mhc', 'bind', 'pt'), edge_label_index),
    edge_label=edge_label,
    batch_size=batch_size,
    edge_label_time=edge_val,
    shuffle=True
)
meta = data.metadata()
if args.model == 'cnn':
    model = ModelattnFuse(train_data, hidden_channels=args.hidden_dim)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100)
val_best = 1e100
bundle = []
epoch = 500
early_stop = 0
test_best = 0
for epoch in tqdm.tqdm(range(1, epoch+1)):
    if early_stop >= 20: break
    total_loss = total_examples = 0
    total_loss_reg = 0
    model.train()
    for i in range(1):
        for sampled_data in (train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred, emb = model(sampled_data)
            ground_truth = sampled_data['mhc', 'bind', 'pt'].edge_label
            weight = torch.Tensor([scale]).to(device)
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth, pos_weight=weight)
            loss2 = F.huber_loss(pred, sampled_data['mhc', 'bind', 'pt'].edge_label_time)
            loss_total = loss + 2*loss2
            loss_total.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_loss_reg += float(loss2) * pred.numel()
            total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}, adj Loss: {total_loss_reg / total_examples:.4f}")
    model.eval()
    val_loss, val_auroc, val_auprc = validate(val_data, model)
    if epoch >= 0:
        scheduler.step(val_loss)
        if val_loss < val_best:
            early_stop = 0
            val_best = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'ckpt/ckpt_{args.model}_{args.seed}_{args.disjoint_rate}.pth')
            print(f"Validation - AUROC: {val_auroc:.4f}; AUPRC: {val_auprc:.4f}; Loss: {val_loss:.4f}")
        else: 
            early_stop += 1
        

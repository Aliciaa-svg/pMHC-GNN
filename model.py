import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import to_hetero, Linear, TransformerConv
import math
from torch_geometric.nn import HeteroBatchNorm, BatchNorm
from tape import ProteinResNetModel

class GNNattn(torch.nn.Module):
    def __init__(self, hidden_channels, meta):
        super().__init__()
        self.conv1 = TransformerConv(hidden_channels, hidden_channels, heads=2, concat=False)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=2, concat=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(-1, hidden_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr) + self.lin1(x)
        x = self.conv2(x, edge_index, edge_attr) + self.lin2(x)
        return x


def output_size(in_size, kernel_size, stride, padding):
    return (in_size + 2*padding - kernel_size) // stride + 1

class CNN(torch.nn.Module):
    def __init__(self, n, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=2)
        self.act = nn.ELU()
        self.dropout = nn.Dropout()
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        conv1_out = output_size(n, 3, 1, 2)
        pool_out1 = output_size(conv1_out, 3, 1, 0)
        conv2_out = output_size(pool_out1, 3, 1, 2)
        pool_out2 = output_size(conv2_out, 3, 1, 0)
        self.fc1 = nn.Linear(hidden_dim * pool_out2, output_dim)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.act(self.batchnorm(x))
        x = self.dropout(x)
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.shape[0], -1)  # Flatten the output
        x = self.bn2(self.fc1(x))
        return x


class ProteinLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    def forward(self, inputs):
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        return self.lstm(inputs)

class ProteinLSTMPooler(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers=1):
        super().__init__()
        self.scalar_reweighting = nn.Linear(2 * num_hidden_layers, 1)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.scalar_reweighting(hidden_states).squeeze(2)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ProteinLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers=1, hidden_dropout_prob=0.1):
        super().__init__()
        forward_lstm = [ProteinLSTMLayer(input_size, hidden_size)]
        reverse_lstm = [ProteinLSTMLayer(input_size, hidden_size)]
        for _ in range(num_hidden_layers - 1):
            forward_lstm.append(ProteinLSTMLayer(
                hidden_size, hidden_size, hidden_dropout_prob))
            reverse_lstm.append(ProteinLSTMLayer(
                hidden_size, hidden_size, hidden_dropout_prob))
        self.forward_lstm = nn.ModuleList(forward_lstm)
        self.reverse_lstm = nn.ModuleList(reverse_lstm)
        # self.output_hidden_states = config.output_hidden_states
        self.output_hidden_states = None
    
    def forward(self, inputs, input_mask=None):
        all_forward_pooled = ()
        all_reverse_pooled = ()
        all_hidden_states = (inputs,)
        forward_output = inputs
        for layer in self.forward_lstm:
            forward_output, forward_pooled = layer(forward_output)
            all_forward_pooled = all_forward_pooled + (forward_pooled[0],)
            all_hidden_states = all_hidden_states + (forward_output,)

        reversed_sequence = self.reverse_sequence(inputs, input_mask)
        reverse_output = reversed_sequence
        for layer in self.reverse_lstm:
            reverse_output, reverse_pooled = layer(reverse_output)
            all_reverse_pooled = all_reverse_pooled + (reverse_pooled[0],)
            all_hidden_states = all_hidden_states + (reverse_output,)
        reverse_output = self.reverse_sequence(reverse_output, input_mask)

        output = torch.cat((forward_output, reverse_output), dim=2)
        pooled = all_forward_pooled + all_reverse_pooled
        pooled = torch.stack(pooled, 3).squeeze(0)
        outputs = (output, pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        return outputs  # sequence_embedding, pooled_embedding, (hidden_states)
    
    def reverse_sequence(self, sequence, input_mask):
        if input_mask is None:
            idx = torch.arange(sequence.size(1) - 1, -1, -1)
            idx = idx.to(sequence.device)
            reversed_sequence = sequence.index_select(1, idx)
        else:
            sequence_lengths = input_mask.sum(1)
            reversed_sequence = []
            for seq, seqlen in zip(sequence, sequence_lengths):
                idx = torch.arange(seqlen - 1, -1, -1, device=seq.device)
                seq = seq.index_select(0, idx)
                seq = F.pad(seq, [0, 0, 0, sequence.size(1) - seqlen])
                reversed_sequence.append(seq)
            reversed_sequence = torch.stack(reversed_sequence, 0)
        return reversed_sequence


class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x1, x2, edge_label_index):
        edge_feat_1 = x1[edge_label_index[0]]
        edge_feat_2 = x2[edge_label_index[1]]        
        x = torch.cat([edge_feat_1, edge_feat_2], dim=-1)
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class ModelattnFuse(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        self.cnn1 = CNN(data["pt"].x.shape[2], data["pt"].x.shape[1], hidden_channels, hidden_channels)
        self.cnn2 = CNN(data["mhc"].x.shape[2], data["mhc"].x.shape[1], hidden_channels, hidden_channels)
        self.gnn = GNNattn(hidden_channels, data.metadata())
        meta1 = (['mhc', 'pt'], [('mhc', 'bind', 'pt'), ('pt', 'rev_bind', 'mhc')])
        self.gnn = to_hetero(self.gnn, metadata=meta1)
        self.classifier = Classifier(2*hidden_channels)
        self.hidden = hidden_channels
        meta2 = (['mhc'], [('mhc', 'mm', 'mhc')])
        meta3 = (['pt'], [('pt', 'pp', 'pt')])
        self.gnn_m = GNNattn(hidden_channels, data.metadata())
        self.gnn_p = GNNattn(hidden_channels, data.metadata())
        self.gnn_m = to_hetero(self.gnn_m, metadata=meta2)
        self.gnn_p = to_hetero(self.gnn_p, metadata=meta3)
        self.lin1 = nn.Linear(2*hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(2*hidden_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        
    def forward(self, data):
        b = 1000
        data_mhc = torch.zeros((data["mhc"].x.shape[0], self.hidden), device=data["mhc"].x.device)
        data_pt = torch.zeros((data["pt"].x.shape[0], self.hidden), device=data["pt"].x.device)
        for i in (range(math.ceil(data["mhc"].x.shape[0]/b))):
            data_mhc[i*b:i*b+b] = self.cnn2(data["mhc"].x[i*b:i*b+b])
        for i in (range(math.ceil(data["pt"].x.shape[0]/b))):
            data_pt[i*b:i*b+b] = self.cnn1(data["pt"].x[i*b:i*b+b])
            torch.cuda.empty_cache()
        x_dict = {
          "mhc": (data_mhc),
          "pt": (data_pt),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        x_m = self.gnn_m(x_dict, data.edge_index_dict)
        x_p = self.gnn_p(x_dict, data.edge_index_dict)
        x_dict['mhc'] = ((torch.cat([x_dict['mhc'], x_m['mhc']], dim=1)))
        x_dict['pt'] = ((torch.cat([x_dict['pt'], x_p['pt']], dim=1)))
        pred = self.classifier(
            x_dict["mhc"],
            x_dict["pt"],
            data['mhc', 'bind', 'pt'].edge_label_index,
        )
        return pred.squeeze(1), x_dict


class Modelattn(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        self.cnn1 = CNN(data["pt"].x.shape[2], data["pt"].x.shape[1], hidden_channels, hidden_channels)
        self.cnn2 = CNN(data["mhc"].x.shape[2], data["mhc"].x.shape[1], hidden_channels, hidden_channels)
        self.gnn = GNNattn(hidden_channels, data.metadata())
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())         
        self.classifier = Classifier(hidden_channels)
        self.hidden = hidden_channels
        self.lin1 = Linear(data["mhc"].x.shape[1]*data["mhc"].x.shape[2], hidden_channels)
        self.lin2 = Linear(data["pt"].x.shape[1]*data["pt"].x.shape[2], hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
    
    def forward(self, data):
        b = 2000
        data_mhc = torch.zeros((data["mhc"].x.shape[0], self.hidden), device=data["mhc"].x.device)
        data_pt = torch.zeros((data["pt"].x.shape[0], self.hidden), device=data["pt"].x.device)
        for i in (range(math.ceil(data["mhc"].x.shape[0]/b))):
            data_mhc[i*b:i*b+b] = self.cnn2(data["mhc"].x[i*b:i*b+b])
        for i in (range(math.ceil(data["pt"].x.shape[0]/b))):
            data_pt[i*b:i*b+b] = self.cnn1(data["pt"].x[i*b:i*b+b])
            torch.cuda.empty_cache()
        x_dict = {
          "mhc": (data_mhc),
          "pt": (data_pt),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["mhc"],
            x_dict["pt"],
            data['mhc', 'bind', 'pt'].edge_label_index,
        )
        return pred.squeeze(1), x_dict
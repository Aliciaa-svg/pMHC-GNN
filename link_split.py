import copy
import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType
from torch_geometric.utils import negative_sampling


@functional_transform('link_split_my')
class RandomLinkSplitNeg(BaseTransform):
    
    def __init__(
        self,
        edge_neg,
        num_val: Union[int, float] = 0.1,
        num_test: Union[int, float] = 0.2,
        is_undirected: bool = False,
        key: str = 'edge_label',
        split_labels: bool = False,
        add_negative_train_samples: bool = True,
        neg_sampling_ratio: float = 1.0,
        disjoint_train_ratio: Union[int, float] = 0.0,
        edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
        rev_edge_types: Optional[Union[
            EdgeType,
            List[Optional[EdgeType]],
        ]] = None,
    ) -> None:
        if isinstance(edge_types, list):
            if rev_edge_types is None:
                rev_edge_types = [None] * len(edge_types)

            assert isinstance(rev_edge_types, list)
            assert len(edge_types) == len(rev_edge_types)

        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.key = key
        self.split_labels = split_labels
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio
        self.disjoint_train_ratio = disjoint_train_ratio
        self.edge_types = edge_types
        self.rev_edge_types = rev_edge_types
        self.edge_neg = edge_neg

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Tuple[
            Union[Data, HeteroData],
            Union[Data, HeteroData],
            Union[Data, HeteroData],
    ]:
        edge_types = self.edge_types
        rev_edge_types = self.rev_edge_types

        train_data = copy.copy(data)
        val_data = copy.copy(data)
        test_data = copy.copy(data)

        num_nodes = data['mhc'].x.shape[0]

        if isinstance(data, HeteroData):
            assert isinstance(train_data, HeteroData)
            assert isinstance(val_data, HeteroData)
            assert isinstance(test_data, HeteroData)

            if edge_types is None:
                raise ValueError(
                    "The 'RandomLinkSplit' transform expects 'edge_types' to "
                    "be specified when operating on 'HeteroData' objects")

            if not isinstance(edge_types, list):
                assert not isinstance(rev_edge_types, list)
                edge_types = [edge_types]
                rev_edge_types = [rev_edge_types]

            stores = [data[edge_type] for edge_type in edge_types]
            train_stores = [train_data[edge_type] for edge_type in edge_types]
            val_stores = [val_data[edge_type] for edge_type in edge_types]
            test_stores = [test_data[edge_type] for edge_type in edge_types]
        else:
            assert isinstance(train_data, Data)
            assert isinstance(val_data, Data)
            assert isinstance(test_data, Data)

            rev_edge_types = [None]

            train_data = copy.copy(data)
            val_data = copy.copy(data)
            test_data = copy.copy(data)

            stores = [data._store]
            train_stores = [train_data._store]
            val_stores = [val_data._store]
            test_stores = [test_data._store]

        assert isinstance(rev_edge_types, list)
        for item in zip(stores, train_stores, val_stores, test_stores,
                        rev_edge_types):
            store, train_store, val_store, test_store, rev_edge_type = item
            
            is_undirected = self.is_undirected
            is_undirected &= not store.is_bipartite()
            is_undirected &= (rev_edge_type is None
                              or (isinstance(data, HeteroData)
                                  and store._key == data[rev_edge_type]._key))

            edge_index = store.edge_index
            
            perm = torch.randperm(num_nodes)
            num_val = self.num_val
            if isinstance(num_val, float):
                num_val = int(num_val * perm.numel())
            num_test = self.num_test
            if isinstance(num_test, float):
                num_test = int(num_test * perm.numel())
            num_train = perm.numel() - num_val - num_test
            if num_train <= 0:
                raise ValueError("Insufficient number of edges for training")
            
            train_node = perm[:num_train]
            val_node = perm[num_train:num_train + num_val]
            test_node = perm[num_train + num_val:]
            mask_train = torch.isin(edge_index[0], train_node)
            mask_val = torch.isin(edge_index[0], val_node)
            mask_test = torch.isin(edge_index[0], test_node)

            mask_train_neg = torch.isin(self.edge_neg[0], train_node)
            mask_val_neg = torch.isin(self.edge_neg[0], val_node)
            mask_test_neg = torch.isin(self.edge_neg[0], test_node)
            
            if is_undirected:
                mask = edge_index[0] <= edge_index[1]
                perm = mask.nonzero(as_tuple=False).view(-1)
                perm = perm[torch.randperm(perm.size(0), device=perm.device)]
            else:
                device = edge_index.device
                perm = torch.randperm(edge_index.size(1), device=device)
                # perm = torch.arange(edge_index.size(1), device=device)

            num_val = self.num_val
            if isinstance(num_val, float):
                num_val = int(num_val * perm.numel())
            num_test = self.num_test
            if isinstance(num_test, float):
                num_test = int(num_test * perm.numel())
            num_train = perm.numel() - num_val - num_test
            train_edges = perm[:num_train]
            val_edges = perm[num_train:num_train + num_val]
            test_edges = perm[num_train + num_val:]
            train_val_edges = perm[:num_train + num_val]
            

            num_train = train_edges.numel()
            num_val = val_edges.numel()
            num_test = test_edges.numel()
            num_disjoint = self.disjoint_train_ratio
            if isinstance(num_disjoint, float):
                num_disjoint = int(num_disjoint * train_edges.numel())
            if num_train - num_disjoint <= 0:
                raise ValueError("Insufficient number of edges for training")

            # Create data splits:
            self._split(train_store, train_edges[num_disjoint:], is_undirected,
                        rev_edge_type)
            self._split(val_store, train_edges, is_undirected, rev_edge_type)
            self._split(test_store, train_val_edges, is_undirected,
                        rev_edge_type)

            num_neg_total = self.edge_neg.size(1)
            perm_neg = torch.randperm(self.edge_neg.size(1), device=device)
            num_val_neg = self.num_val
            if isinstance(num_val_neg, float):
                num_val_neg = int(num_val_neg * perm_neg.numel())
            num_test_neg = self.num_test
            if isinstance(num_test, float):
                num_test_neg = int(num_test_neg * perm_neg.numel())

            num_train_neg = int((perm_neg.numel() - num_val_neg - num_test_neg))
            
            c=1
            c0=1
            train_edges_neg = perm_neg[:int(num_train_neg*c)]
            val_edges_neg = perm_neg[int(num_train_neg*c): int(num_train_neg*c)+int(num_val_neg*c0)]
            test_edges_neg = perm_neg[int(num_train_neg*c)+int(num_val_neg*c0):]

            num_neg_train = 0
            if self.add_negative_train_samples:
                if num_disjoint > 0:
                    num_neg_train = int(num_disjoint * self.neg_sampling_ratio)
                else:
                    num_neg_train = int(num_train * self.neg_sampling_ratio)
            num_neg_val = int(num_val * self.neg_sampling_ratio)
            num_neg_test = int(num_test * self.neg_sampling_ratio)

            num_neg = num_neg_train + num_neg_val + num_neg_test

            size = store.size()
            if store._key is None or store._key[0] == store._key[-1]:
                size = size[0]
            neg_edge_index = negative_sampling(edge_index, size,
                                               num_neg_samples=num_neg,
                                               method='sparse')

            # Adjust ratio if not enough negative edges exist
            if neg_edge_index.size(1) < num_neg:
                num_neg_found = neg_edge_index.size(1)
                ratio = num_neg_found / num_neg
                warnings.warn(
                    f"There are not enough negative edges to satisfy "
                    "the provided sampling ratio. The ratio will be "
                    f"adjusted to {ratio:.2f}.")
                num_neg_train = int((num_neg_train / num_neg) * num_neg_found)
                num_neg_val = int((num_neg_val / num_neg) * num_neg_found)
                num_neg_test = num_neg_found - num_neg_train - num_neg_val

            # Create labels:
            if num_disjoint > 0:
                train_edges = train_edges[:int(num_disjoint*1.5)]
            self._create_label(
                store,
                train_edges,
                self.edge_neg,
                train_edges_neg,
                out=train_store,
            )
            self._create_label(
                store,
                val_edges,
                self.edge_neg,
                val_edges_neg,
                out=val_store,
            )
            self._create_label(
                store,
                test_edges,
                self.edge_neg,
                test_edges_neg,
                out=test_store,
            )

        return train_data, val_data, test_data

    def _split(
        self,
        store: EdgeStorage,
        index: Tensor,
        is_undirected: bool,
        rev_edge_type: Optional[EdgeType],
    ) -> EdgeStorage:

        edge_attrs = {key for key in store.keys() if store.is_edge_attr(key)}
        for key, value in store.items():
            if key == 'edge_index':
                continue

            if key in edge_attrs:
                value = value[index]
                if is_undirected:
                    value = torch.cat([value, value], dim=0)
                store[key] = value

        edge_index = store.edge_index[:, index]
        if is_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
        store.edge_index = edge_index

        if rev_edge_type is not None:
            rev_store = store._parent()[rev_edge_type]
            for key in rev_store.keys():
                if key not in store:
                    del rev_store[key]  # We delete all outdated attributes.
                elif key == 'edge_index':
                    rev_store.edge_index = store.edge_index.flip([0])
                else:
                    rev_store[key] = store[key]

        return store

    def _create_label(
        self,
        store: EdgeStorage,
        index: Tensor,
        neg_edge_index: Tensor,
        neg_edge: Tensor,
        out: EdgeStorage,
    ) -> EdgeStorage:

        edge_index = store.edge_index[:, index]
        # print(store)
        # print(neg_edge_index.shape)

        if hasattr(store, 'edge_attr'):
            # print(store.edge_attr.shape)
            edge_attr = store.edge_attr[index]
            # print(edge_attr.shape)

            edge_attr_neg_idx = store.edge_attr[-neg_edge_index.size(1):]
            edge_attr_neg = edge_attr_neg_idx[neg_edge]
            # print(edge_attr_neg.shape)
            edge_attr = torch.cat([edge_attr, edge_attr_neg])
        
        neg_edge_index = neg_edge_index[:, neg_edge]
        if hasattr(store, self.key):
            edge_label = store[self.key]
            edge_label = edge_label[index]
            # Increment labels by one. Note that there is no need to increment
            # in case no negative edges are added.
            if neg_edge_index.numel() > 0:
                assert edge_label.dtype == torch.long
                assert edge_label.size(0) == edge_index.size(1)
                edge_label.add_(1)
            if hasattr(out, self.key):
                delattr(out, self.key)
        else:
            edge_label = torch.ones(index.numel(), device=index.device)

        if neg_edge_index.numel() > 0:
            neg_edge_label = edge_label.new_zeros((neg_edge_index.size(1), ) +
                                                  edge_label.size()[1:])

        if self.split_labels:
            out[f'pos_{self.key}'] = edge_label
            out[f'pos_{self.key}_index'] = edge_index
            if neg_edge_index.numel() > 0:
                out[f'neg_{self.key}'] = neg_edge_label
                out[f'neg_{self.key}_index'] = neg_edge_index

        else:
            if neg_edge_index.numel() > 0:
                edge_label = torch.cat([edge_label, neg_edge_label], dim=0)
                edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)
            out[self.key] = edge_label
            out[f'{self.key}_index'] = edge_index
        if hasattr(store, 'edge_attr'):
            out['edge_attr'] = edge_attr

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_val={self.num_val}, '
                f'num_test={self.num_test})')
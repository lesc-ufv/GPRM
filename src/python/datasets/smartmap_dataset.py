import torch
from torch_geometric.data import Data, Batch
from src.python.ddp_launcher import Launcher
from src.python.datasets.target_dataset import TargetDataset
class SmartMapDataset(torch.utils.data.Dataset):

    def __init__(self, model_inputs, targets:dict):
        all_tensors = model_inputs

        self.target_data = TargetDataset(targets)
        
        self.max_action_size = all_tensors[-1]
        self.max_cgra_size = all_tensors[-2]
        self.max_dfg_size = all_tensors[-3]
        self.batch_size = all_tensors[-4].item()

        self.dfg_features = all_tensors[0:self.batch_size]
        self.dfg_edge_indices = all_tensors[self.batch_size:2*self.batch_size]
        self.cgra_features = all_tensors[2*self.batch_size:3*self.batch_size]
        self.cgra_edge_indices = all_tensors[3*self.batch_size:4*self.batch_size]
        self.node_to_be_mapped_features = all_tensors[4*self.batch_size:5*self.batch_size]
        self.actions_features = all_tensors[5*self.batch_size:6*self.batch_size]
        self.actions_masks = all_tensors[6*self.batch_size:7*self.batch_size]
        self.dfg_masks = all_tensors[7*self.batch_size:8*self.batch_size]
        self.cgra_masks = all_tensors[8*self.batch_size:9*self.batch_size]

    def __len__(self):
        return len(self.dfg_features)

    def __getitem__(self, index):
        dfg_features = self.dfg_features[index]
        dfg_edge_indices = self.dfg_edge_indices[index]
        cgra_features = self.cgra_features[index]
        cgra_edge_indices = self.cgra_edge_indices[index]
        node_to_be_mapped_features = self.node_to_be_mapped_features[index]
        actions_features = self.actions_features[index]
        actions_masks = self.actions_masks[index]
        dfg_masks = self.dfg_masks[index]
        cgra_masks = self.cgra_masks[index]

        return {"model_inputs":[dfg_features, dfg_edge_indices,
                cgra_features, cgra_edge_indices,
                node_to_be_mapped_features,
                actions_features,
                actions_masks,
                dfg_masks,
                cgra_masks],
                "targets":self.target_data.__getitem__(index)}
        
    @staticmethod
    def add_and_pad_graph(graph_features, graph_pad_mask, max_graph_size):
        graph_size = graph_features.size(0)
        if graph_size < max_graph_size:
            diff = max_graph_size - graph_size
            graph_features = torch.cat([graph_features, torch.zeros((diff, graph_features.size(1)))] ,0)
            graph_pad_mask = torch.cat([graph_pad_mask, torch.zeros((1, diff))], 1)
        return graph_features, graph_pad_mask

    def collate_fn(self, samples):
        batch_dfg = []
        batch_cgra = []
        batch_node_to_be_mapped_features = []
        batch_actions_features = []
        batch_actions_mask_pad = []
        batch_dfg_mask_pad = []
        batch_cgra_mask_pad = []

        max_dfg_size = self.max_dfg_size
        max_cgra_size = self.max_cgra_size
        max_action_size = self.max_action_size
        targets = []
        for sample in samples:
            dfg_features, dfg_edge_indices, cgra_features, cgra_edge_indices, \
            node_to_be_mapped_features, actions_features, actions_masks, dfg_masks, cgra_masks = sample["model_inputs"]
            targets.append(sample["targets"])
            
            new_dfg_features, new_dfg_mask = self.add_and_pad_graph(dfg_features, dfg_masks, max_dfg_size)
            dfg_data = Data(x=new_dfg_features, edge_index=dfg_edge_indices)
            batch_dfg_mask_pad.append(new_dfg_mask)
            batch_dfg.append(dfg_data)

            new_cgra_features, new_cgra_mask = self.add_and_pad_graph(cgra_features, cgra_masks, max_cgra_size)
            cgra_data = Data(x=new_cgra_features, edge_index=cgra_edge_indices)
            batch_cgra_mask_pad.append(new_cgra_mask)

            batch_cgra.append(cgra_data)

            batch_node_to_be_mapped_features.append(node_to_be_mapped_features)

            actions_features_tensor = actions_features
            actions_mask_tensor = actions_masks
            cur_actions_size = actions_features_tensor.size(1)

            if cur_actions_size < max_action_size:
                diff = max_action_size - cur_actions_size
                actions_features_tensor = torch.cat([
                    actions_features_tensor,
                    torch.zeros((1, diff, actions_features_tensor.size(-1)), dtype=torch.float32)
                ], 1)

                actions_mask_tensor = torch.cat([
                    actions_mask_tensor,
                    torch.zeros((1, diff), dtype=torch.bool)
                ], -1)

            batch_actions_features.append(actions_features_tensor)
            batch_actions_mask_pad.append(actions_mask_tensor)

        batch_dfg = Batch.from_data_list(batch_dfg)
        batch_cgra = Batch.from_data_list(batch_cgra)

        batch_actions_features = torch.cat(batch_actions_features,0)
        batch_actions_mask_pad = torch.cat(batch_actions_mask_pad,0)
        batch_node_to_be_mapped_features = torch.cat(batch_node_to_be_mapped_features,0)
        batch_dfg_mask_pad = torch.cat(batch_dfg_mask_pad,0)
        batch_cgra_mask_pad = torch.cat(batch_cgra_mask_pad,0)

        targets = self.target_data.collate_fn(targets)

        return {"model_inputs":[batch_dfg.x, batch_dfg.edge_index, batch_cgra.x, batch_cgra.edge_index, batch_node_to_be_mapped_features, batch_actions_features, \
            batch_actions_mask_pad, batch_dfg_mask_pad, batch_cgra_mask_pad ], "targets": targets }

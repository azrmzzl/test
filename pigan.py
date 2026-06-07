import glob
import os
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


SEED = 2025
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
BATCH_SIZE = 32
EPOCHS = 100
MASK_RATE = 0.15
HINT_RATE = 0.9
ALPHA = 10.0
BETA = 2.0
PRINT_EVERY = 5
BEST_MODEL_GLOB = "pg_graphgan_G_best_*.pth"
LATEST_MODEL_PATH = "pg_graphgan_G_pretrained.pth"
FEATURE_NAMES = ["feature_0", "feature_1", "feature_2", "feature_3"]


def set_random_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IEEE57GraphDataset:
    def __init__(self, data_dir="dataset_split"):
        print(f">>> Loading IEEE 57 dataset from: {data_dir}")

        self.samples = sio.loadmat(os.path.join(data_dir, "Samples_x_y_normal_operation_all.mat"))["Samples"]
        self.branch_data = sio.loadmat(os.path.join(data_dir, "Topology_Branch.mat"))["branch_data"]

        pmu_raw = sio.loadmat(os.path.join(data_dir, "PMU_Position.mat"))["pmu_position"][0]
        self.pmu_pos = pmu_raw.astype(np.int64) - 1
        self.branch_data = self.branch_data.astype(np.int64) - 1

        self.num_buses = 57
        self.num_lines = self.branch_data.shape[0]
        self.num_nodes = self.num_buses + self.num_lines

        self.edge_index, self.B_inc = self._build_graph_structure()
        self.x = self._process_features()

        print(f"Dataset tensor shape: {tuple(self.x.shape)}")

    def _build_graph_structure(self):
        adj = torch.eye(self.num_nodes)
        b_inc = torch.zeros(self.num_buses, self.num_lines)

        for line_idx in range(self.num_lines):
            bus_from = int(self.branch_data[line_idx, 0])
            bus_to = int(self.branch_data[line_idx, 1])
            line_node = self.num_buses + line_idx

            adj[bus_from, line_node] = 1
            adj[line_node, bus_from] = 1
            adj[bus_to, line_node] = 1
            adj[line_node, bus_to] = 1

            b_inc[bus_from, line_idx] = 1.0
            b_inc[bus_to, line_idx] = -1.0

        edge_index, _ = dense_to_sparse(adj)
        return edge_index, b_inc

    def _process_features(self):
        num_samples = self.samples.shape[0]
        x_graph = torch.zeros(num_samples, self.num_nodes, 4)

        idx = 0
        idx_v_scada = slice(0, 57)
        idx += 57
        idx_p_inj = slice(idx, idx + 57)
        idx += 57
        idx_q_inj = slice(idx, idx + 57)
        idx += 57
        idx_p_flow = slice(idx, idx + 80)
        idx += 80
        idx_q_flow = slice(idx, idx + 80)
        idx += 80
        n_pmu = len(self.pmu_pos)
        idx_del_pmu = slice(idx, idx + n_pmu)
        idx += n_pmu
        idx_v_pmu = slice(idx, idx + n_pmu)
        idx += n_pmu

        i_from_mask = np.isin(self.branch_data[:, 0], self.pmu_pos)
        i_to_mask = np.isin(self.branch_data[:, 1], self.pmu_pos)
        n_if = np.sum(i_from_mask)
        n_it = np.sum(i_to_mask)
        idx_ire_from = slice(idx, idx + n_if)
        idx += n_if
        idx_iim_from = slice(idx, idx + n_if)
        idx += n_if
        idx_ire_to = slice(idx, idx + n_it)
        idx += n_it
        idx_iim_to = slice(idx, idx + n_it)
        idx += n_it

        samples_t = torch.tensor(self.samples, dtype=torch.float32)

        x_graph[:, :57, 0] = samples_t[:, idx_v_scada]
        x_graph[:, :57, 1] = samples_t[:, idx_p_inj]
        x_graph[:, :57, 2] = samples_t[:, idx_q_inj]
        x_graph[:, self.pmu_pos, 3] = samples_t[:, idx_del_pmu]

        x_graph[:, 57:, 0] = samples_t[:, idx_p_flow]
        x_graph[:, 57:, 1] = samples_t[:, idx_q_flow]

        branch_indices_from = np.where(i_from_mask)[0]
        branch_indices_to = np.where(i_to_mask)[0]
        x_graph[:, 57 + branch_indices_from, 2] = samples_t[:, idx_ire_from]
        x_graph[:, 57 + branch_indices_from, 3] = samples_t[:, idx_iim_from]
        x_graph[:, 57 + branch_indices_to, 2] = samples_t[:, idx_ire_to]
        x_graph[:, 57 + branch_indices_to, 3] = samples_t[:, idx_iim_to]

        return x_graph


def physics_loss_ieee57_direct(recon_x, b_inc):
    num_buses = 57
    bus_p = recon_x[:, :num_buses, 1]
    bus_q = recon_x[:, :num_buses, 2]
    branch_p = recon_x[:, num_buses:, 0]
    branch_q = recon_x[:, num_buses:, 1]

    agg_p = torch.matmul(branch_p, b_inc.T)
    agg_q = torch.matmul(branch_q, b_inc.T)

    loss_p = F.mse_loss(bus_p, agg_p)
    loss_q = F.mse_loss(bus_q, agg_q)
    return loss_p + loss_q


class PG_GraphGenerator(nn.Module):
    def __init__(self, in_channels=4, hidden=64, out_channels=4, heads=2):
        super().__init__()
        self.out_channels = out_channels
        self.gat1 = GATConv(in_channels + 1, hidden, heads=heads, concat=True)
        self.gat2 = GATConv(hidden * heads, out_channels, heads=1, concat=False)

    def forward(self, x_corrupted, node_mask, edge_batch):
        batch_size, num_nodes, _ = x_corrupted.shape
        inputs = torch.cat([x_corrupted, node_mask], dim=-1)
        inputs_flat = inputs.reshape(batch_size * num_nodes, -1)
        hidden = F.elu(self.gat1(inputs_flat, edge_batch))
        output = self.gat2(hidden, edge_batch)
        return output.reshape(batch_size, num_nodes, self.out_channels)


class PG_GraphDiscriminator(nn.Module):
    def __init__(self, in_channels=4, hidden=64, heads=2):
        super().__init__()
        self.gat1 = GATConv(in_channels + 1, hidden, heads=heads, concat=True)
        self.gat2 = GATConv(hidden * heads, 1, heads=1, concat=False)

    def forward(self, x_imputed, hint, edge_batch):
        batch_size, num_nodes, _ = x_imputed.shape
        inputs = torch.cat([x_imputed, hint], dim=-1)
        inputs_flat = inputs.reshape(batch_size * num_nodes, -1)
        hidden = F.elu(self.gat1(inputs_flat, edge_batch))
        logits = self.gat2(hidden, edge_batch)
        return torch.sigmoid(logits).reshape(batch_size, num_nodes, 1)


def sample_hint(mask, hint_rate=HINT_RATE):
    batch_size, num_nodes, num_dims = mask.shape
    hint_mask = (torch.rand(batch_size, num_nodes, num_dims, device=mask.device) < hint_rate).float()
    return mask * hint_mask + 0.5 * (1 - hint_mask)


def split_dataset(x_all, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    num_samples = x_all.shape[0]
    if num_samples < 3:
        raise ValueError("At least 3 samples are required to build train/val/test splits.")

    num_train = max(1, int(num_samples * train_ratio))
    num_val = max(1, int(num_samples * val_ratio))
    num_test = num_samples - num_train - num_val

    while num_test < 1:
        if num_train > num_val and num_train > 1:
            num_train -= 1
        elif num_val > 1:
            num_val -= 1
        else:
            raise ValueError("Unable to create a non-empty test split.")
        num_test = num_samples - num_train - num_val

    train_data = x_all[:num_train]
    val_data = x_all[num_train:num_train + num_val]
    test_data = x_all[num_train + num_val:]
    return train_data, val_data, test_data


def build_edge_batch(edge_index, batch_size, num_nodes, device):
    num_edges = edge_index.shape[1]
    offsets = (torch.arange(batch_size, device=device) * num_nodes).view(batch_size, 1, 1)
    return (edge_index.view(1, 2, num_edges) + offsets).reshape(2, batch_size * num_edges)


def create_corrupted_input(batch_x, mask_rate=MASK_RATE):
    batch_size, num_nodes, num_features = batch_x.shape
    node_mask = (torch.rand(batch_size, num_nodes, 1, device=batch_x.device) > mask_rate).float()
    feature_mask = node_mask.expand(batch_size, num_nodes, num_features)

    noise_scale = batch_x.std(unbiased=False).clamp_min(1e-6)
    noise_shift = batch_x.mean()
    noise = torch.rand(batch_size, num_nodes, num_features, device=batch_x.device) * noise_scale + noise_shift

    x_corrupted = batch_x * feature_mask + noise * (1 - feature_mask)
    return node_mask, feature_mask, x_corrupted


def masked_mse(pred, target, mask):
    denom = mask.sum().clamp_min(1.0)
    return torch.sum(((pred - target) ** 2) * mask) / denom


def evaluate_generator(net_g, data_loader, edge_index, b_inc, device, mask_rate=MASK_RATE):
    net_g.eval()

    total_masked_sse = 0.0
    total_masked_sae = 0.0
    total_masked_count = 0.0
    total_all_sse = 0.0
    total_all_count = 0.0
    total_phy = 0.0
    num_batches = 0

    feature_sse = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    feature_sae = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    feature_count = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x.to(device)
            batch_size, num_nodes, num_features = batch_x.shape

            edge_batch = build_edge_batch(edge_index, batch_size, num_nodes, device)
            node_mask, feature_mask, x_corrupted = create_corrupted_input(batch_x, mask_rate)
            missing_mask = 1 - feature_mask

            x_generated = net_g(x_corrupted, node_mask, edge_batch)
            x_imputed = batch_x * feature_mask + x_generated * missing_mask
            diff = x_imputed - batch_x

            total_masked_sse += torch.sum((diff ** 2) * missing_mask).item()
            total_masked_sae += torch.sum(diff.abs() * missing_mask).item()
            total_masked_count += missing_mask.sum().item()
            total_all_sse += torch.sum(diff ** 2).item()
            total_all_count += diff.numel()
            total_phy += physics_loss_ieee57_direct(x_imputed, b_inc).item()
            num_batches += 1

            for feat_idx in range(num_features):
                feat_mask = missing_mask[:, :, feat_idx]
                feat_diff = diff[:, :, feat_idx]
                feature_sse[feat_idx] += torch.sum((feat_diff ** 2) * feat_mask).item()
                feature_sae[feat_idx] += torch.sum(feat_diff.abs() * feat_mask).item()
                feature_count[feat_idx] += feat_mask.sum().item()

    masked_mse_value = total_masked_sse / max(total_masked_count, 1.0)
    all_mse_value = total_all_sse / max(total_all_count, 1.0)

    feature_metrics = {}
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        count = max(feature_count[feat_idx], 1.0)
        feat_mse = feature_sse[feat_idx] / count
        feat_mae = feature_sae[feat_idx] / count
        feature_metrics[feat_name] = {
            "mse": feat_mse,
            "rmse": float(np.sqrt(feat_mse)),
            "mae": feat_mae,
        }

    return {
        "masked_mse": masked_mse_value,
        "masked_rmse": float(np.sqrt(masked_mse_value)),
        "masked_mae": total_masked_sae / max(total_masked_count, 1.0),
        "all_mse": all_mse_value,
        "all_rmse": float(np.sqrt(all_mse_value)),
        "physics_loss": total_phy / max(num_batches, 1),
        "feature_metrics": feature_metrics,
    }


def main_train():
    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = IEEE57GraphDataset(data_dir="dataset_split")
    x_all = dataset.x
    edge_index = dataset.edge_index.to(device)
    b_inc = dataset.B_inc.to(device)

    train_data, val_data, test_data = split_dataset(x_all)
    print(
        f"Split sizes -> train: {train_data.shape[0]}, "
        f"val: {val_data.shape[0]}, test: {test_data.shape[0]}"
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    net_g = PG_GraphGenerator().to(device)
    net_d = PG_GraphDiscriminator().to(device)

    opt_g = optim.Adam(net_g.parameters(), lr=1e-3)
    opt_d = optim.Adam(net_d.parameters(), lr=1e-3)

    best_val_score = float("inf")
    best_model_state = None
    best_model_path = None

    print(">>> Start training PG-GraphGAN with validation...")

    for epoch in range(EPOCHS):
        net_g.train()
        net_d.train()

        loss_d_log = 0.0
        loss_g_log = 0.0
        loss_phy_log = 0.0
        train_masked_mse_log = 0.0

        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            batch_size, num_nodes, num_features = batch_x.shape

            edge_batch = build_edge_batch(edge_index, batch_size, num_nodes, device)
            node_mask, feature_mask, x_corrupted = create_corrupted_input(batch_x, MASK_RATE)
            missing_mask = 1 - feature_mask
            hint = sample_hint(node_mask)

            x_generated = net_g(x_corrupted, node_mask, edge_batch)
            x_imputed = batch_x * feature_mask + x_generated * missing_mask
            d_prob = net_d(x_imputed, hint, edge_batch)

            loss_g_adv = -torch.mean((1 - node_mask) * torch.log(d_prob + 1e-8))
            loss_g_mse = masked_mse(x_generated, batch_x, feature_mask)
            loss_g_phy = physics_loss_ieee57_direct(x_imputed, b_inc)
            loss_g = loss_g_adv + ALPHA * loss_g_mse + BETA * loss_g_phy

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            d_prob = net_d(x_imputed.detach(), hint, edge_batch)
            loss_d = -torch.mean(
                node_mask * torch.log(d_prob + 1e-8) +
                (1 - node_mask) * torch.log(1 - d_prob + 1e-8)
            )

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            train_masked_mse = masked_mse(x_imputed, batch_x, missing_mask)
            loss_d_log += loss_d.item()
            loss_g_log += loss_g.item()
            loss_phy_log += loss_g_phy.item()
            train_masked_mse_log += train_masked_mse.item()

        val_metrics = evaluate_generator(net_g, val_loader, edge_index, b_inc, device, MASK_RATE)
        val_score = val_metrics["masked_mse"] + BETA * val_metrics["physics_loss"]

        if val_score < best_val_score:
            best_val_score = val_score
            best_model_state = {k: v.detach().cpu().clone() for k, v in net_g.state_dict().items()}
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            best_model_path = f"pg_graphgan_G_best_{timestamp}.pth"

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            num_train_batches = max(len(train_loader), 1)
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] | "
                f"D_Loss: {loss_d_log / num_train_batches:.4f} | "
                f"G_Loss: {loss_g_log / num_train_batches:.4f} | "
                f"Train Masked MSE: {train_masked_mse_log / num_train_batches:.6f} | "
                f"Val Masked MSE: {val_metrics['masked_mse']:.6f} | "
                f"Val RMSE: {val_metrics['masked_rmse']:.6f} | "
                f"Val MAE: {val_metrics['masked_mae']:.6f} | "
                f"Phy: {loss_phy_log / num_train_batches:.6f}"
            )

    if best_model_state is None or best_model_path is None:
        raise RuntimeError("Training finished but no best model was recorded.")

    torch.save(best_model_state, best_model_path)
    torch.save(best_model_state, LATEST_MODEL_PATH)
    print(f">>> Best generator saved to: {best_model_path}")
    print(f">>> Latest generator alias updated: {LATEST_MODEL_PATH}")

    return best_model_path, test_data


def load_latest_model_path():
    model_files = glob.glob(BEST_MODEL_GLOB)
    if model_files:
        return sorted(model_files)[-1]
    if os.path.exists(LATEST_MODEL_PATH):
        return LATEST_MODEL_PATH
    return None


def print_test_metrics(metrics):
    print("\n>>> Test results")
    print(f"Masked MSE  : {metrics['masked_mse']:.6f}")
    print(f"Masked RMSE : {metrics['masked_rmse']:.6f}")
    print(f"Masked MAE  : {metrics['masked_mae']:.6f}")
    print(f"All MSE     : {metrics['all_mse']:.6f}")
    print(f"All RMSE    : {metrics['all_rmse']:.6f}")
    print(f"PhysicsLoss : {metrics['physics_loss']:.6f}")

    print("\n>>> Per-feature masked metrics")
    for feat_name, feat_metrics in metrics["feature_metrics"].items():
        print(
            f"{feat_name:<10} | "
            f"MSE: {feat_metrics['mse']:.6f} | "
            f"RMSE: {feat_metrics['rmse']:.6f} | "
            f"MAE: {feat_metrics['mae']:.6f}"
        )


def main_test(model_path=None, test_data=None):
    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = IEEE57GraphDataset(data_dir="dataset_split")
    edge_index = dataset.edge_index.to(device)
    b_inc = dataset.B_inc.to(device)

    if test_data is None:
        _, _, test_data = split_dataset(dataset.x)

    if model_path is None:
        model_path = load_latest_model_path()

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError("No generator checkpoint found for testing.")

    print(f">>> Loading generator from: {model_path}")
    net_g = PG_GraphGenerator().to(device)
    state_dict = torch.load(model_path, map_location=device)
    net_g.load_state_dict(state_dict)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    metrics = evaluate_generator(net_g, test_loader, edge_index, b_inc, device, MASK_RATE)
    print_test_metrics(metrics)
    return metrics


def main():
    best_model_path, test_data = main_train()
    main_test(model_path=best_model_path, test_data=test_data)


if __name__ == "__main__":
    main()

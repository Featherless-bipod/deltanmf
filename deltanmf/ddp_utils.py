import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import trange
import numpy as np
from .models import NTCOptimizer

class NMFDataset(Dataset):
    def __init__(self, X):
        """
        X: (m, n) or (genes, cells)
        """
        self.X = X
        self.n_samples = X.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[:, idx], idx

def setup_ddp():
    """Initializes DDP from environment variables if torchrun is used, else falls back to single GPU."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        return local_rank, world_size, True
    else:
        # Fallback to single GPU (local_rank 0)
        local_rank = 0
        world_size = 1
        return local_rank, world_size, False

def cleanup_ddp(is_ddp):
    if is_ddp:
        dist.destroy_process_group()

def solve_ntc_regularized_ddp(
    X, k, S_E=None,
    alpha_ntc=0.0,
    init_W=None, init_H=None,
    max_iter=1000, tol=1e-8,
    nonneg="softplus", softplus_beta=10.0,
    normalize_W=False,
    init_fix_scale=False,
    seed=None,
    lr_start=0.01,
    fm_target_ratio=None,
    fm_apply_late=False,
    fm_last_iters=None,
    batch_size=40960
):
    """
    DDP-enabled variant of solve_ntc_regularized_minibatch utilizing SparseAdam for H.
    Should be launched via torchrun, but degrades gracefully to single-GPU otherwise.
    """
    local_rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    if seed is not None:
        import random
        # First use a common seed to ensure all ranks initialize H identically if random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    m, n = X.shape

    # Allocate Dataset globally on CPU (RAM)
    X_tensor_cpu = torch.tensor(X, dtype=torch.float32)

    # Allocate H globally on CPU (RAM) without requires_grad
    if init_H is not None:
        H_tensor_cpu = torch.tensor(init_H, dtype=torch.float32)
    else:
        H_tensor_cpu = torch.rand(k, n, dtype=torch.float32)

    # set seed with local_rank for data sampling randomness
    if seed is not None:
        random.seed(seed + local_rank)
        np.random.seed(seed + local_rank)
        torch.manual_seed(seed + local_rank)
        torch.cuda.manual_seed_all(seed + local_rank)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    dataset = NMFDataset(X_tensor_cpu)
    
    # Split cells among ranks statically to ensure completely disjoint subsets and local isolated moments
    cells_per_rank = n // world_size
    start_idx = local_rank * cells_per_rank
    end_idx = start_idx + cells_per_rank if local_rank != world_size - 1 else n
    
    local_indices = list(range(start_idx, end_idx))
    local_dataset = torch.utils.data.Subset(dataset, local_indices)
    
    dataloader = DataLoader(
        local_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        prefetch_factor=2
    )

    # Initialize W scaling if needed
    if init_W is not None and init_H is not None and init_fix_scale and local_rank == 0:
        W0 = torch.as_tensor(init_W, dtype=torch.float32)
        H0 = torch.as_tensor(init_H, dtype=torch.float32)
        s0 = W0.sum(dim=0, keepdim=True).clamp_min(1e-12)
        W0 = W0 / s0
        H0 = H0 * s0.squeeze(0).unsqueeze(1)
        init_W = W0.cpu().numpy()
        with torch.no_grad():
            H_tensor_cpu.copy_(H0.cpu())

    if is_ddp:
        dist.barrier()
    
    # Broadcast init_W to all ranks if it was scaled
    if is_ddp and init_W is not None:
        init_W_tensor = torch.tensor(init_W, dtype=torch.float32).to(device)
        dist.broadcast(init_W_tensor, src=0)
        init_W = init_W_tensor.cpu().numpy()

    # Model and DDP wrapping
    model = NTCOptimizer(m, k, init_W=init_W).to(device)
    if is_ddp:
        optimizer_model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        optimizer_model = model

    # Optimizer optimizes model parameters (W) completely standardly
    optimizer_W = optim.Adam(optimizer_model.parameters(), lr=lr_start)

    S_E_tensor = L_S_tensor = None
    if S_E is not None:
        D_E = np.diag(np.sum(S_E, axis=1))
        L_S_tensor = torch.tensor(D_E - S_E, dtype=torch.float32).to(device)

    # Global Adam states for H on CPU
    m_H_cpu = torch.zeros_like(H_tensor_cpu)
    v_H_cpu = torch.zeros_like(H_tensor_cpu)
    beta1, beta2 = 0.9, 0.999
    adam_eps = 1e-8

    if nonneg.lower() == "relu":
        def _act(t): return F.relu(t)
    else:
        def _act(t): return F.softplus(t, beta=softplus_beta)

    loss_history = []
    prev_loss = np.inf
    
    pbar = trange(max_iter, desc="DDP NTC Discovery", leave=False) if local_rank == 0 else range(max_iter)
    eps = torch.as_tensor(1e-12, dtype=torch.float32, device=device)

    total_iters = int(max_iter)
    _last = int(fm_last_iters) if (fm_last_iters is not None) else 0
    fm_start_iter = max(0, total_iters - _last) if (fm_apply_late and _last > 0) else 0

    for epoch in pbar:
        epoch_losses = {key: 0.0 for key in ["recon_loss", "fm_loss", "total_loss"]}
        num_batches = 0
        alpha_updated = False

        for X_b, idx in dataloader:
            optimizer_W.zero_grad()
            
            X_b = X_b.to(device, non_blocking=True)
            
            # Local H Tensor Creation
            H_b_device = H_tensor_cpu[:, idx].clone().detach().to(device)
            H_b_device.requires_grad = True

            W_raw, H_b_out = optimizer_model(H_b_device)
            W_eff = _act(W_raw)
            H_eff = _act(H_b_out)

            if normalize_W:
                s = W_eff.sum(dim=0, keepdim=True).clamp_min(eps)
                W_eff = W_eff / s
                H_eff = H_eff * s.squeeze(0).unsqueeze(1)

            recon_loss = torch.mean((X_b - W_eff @ H_eff) ** 2)
            alpha_t = torch.as_tensor(alpha_ntc, dtype=X_b.dtype, device=device)

            if fm_apply_late and (epoch == fm_start_iter) and (not alpha_updated) and (fm_target_ratio is not None) and (L_S_tensor is not None):
                fm_unscaled = torch.trace(W_eff.T @ L_S_tensor @ W_eff) / (m * k)
                ru = float(recon_loss.item())
                fu = float(fm_unscaled.item())
                if fu > 0.0:
                    alpha_ntc = float((fm_target_ratio * ru) / (fu + 1e-12))
                    alpha_t = torch.as_tensor(alpha_ntc, dtype=X_b.dtype, device=device)
                    alpha_updated = True

            fm_loss = torch.zeros([], dtype=X_b.dtype, device=device)
            if (alpha_ntc > 0) and (L_S_tensor is not None) and (epoch >= fm_start_iter):
                batch_frac = float(X_b.shape[1]) / float(n)
                fm_loss = alpha_t * torch.trace(W_eff.T @ L_S_tensor @ W_eff) / (m * k)
                fm_loss = fm_loss * batch_frac

            total_loss.backward()

            optimizer_W.step()
            
            # Custom Dense-Sparse Adam Step for H
            with torch.no_grad():
                g = H_b_device.grad
                t = epoch + 1
                
                m_b = beta1 * m_H_cpu[:, idx].to(device) + (1 - beta1) * g
                v_b = beta2 * v_H_cpu[:, idx].to(device) + (1 - beta2) * (g ** 2)
                
                m_hat = m_b / (1 - beta1 ** t)
                v_hat = v_b / (1 - beta2 ** t)
                
                H_b_device.sub_(lr_start * m_hat / (torch.sqrt(v_hat) + adam_eps))
                
                # Write back to CPU RAM
                H_tensor_cpu[:, idx] = H_b_device.cpu()
                m_H_cpu[:, idx] = m_b.cpu()
                v_H_cpu[:, idx] = v_b.cpu()

            # Accumulate on CPU purely for logging
            epoch_losses["recon_loss"] += float(recon_loss.item())
            epoch_losses["fm_loss"] += float(fm_loss.item())
            epoch_losses["total_loss"] += float(total_loss.item())
            num_batches += 1

        # Average losses across all GPUs for reporting/early stopping
        avg_losses = {k: torch.tensor(v / max(1, num_batches), device=device) for k, v in epoch_losses.items()}
        if is_ddp:
            for k in avg_losses:
                dist.all_reduce(avg_losses[k], op=dist.ReduceOp.AVG)
        for k in avg_losses:
            avg_losses[k] = float(avg_losses[k].item())

        cur_lr = optimizer_W.param_groups[0]["lr"]
        cur = avg_losses["total_loss"]

        loss_history.append({
            "iteration": int(epoch),
            "lr": float(cur_lr),
            "total_loss": float(cur),
            "recon_loss": float(avg_losses["recon_loss"]),
            "fm_loss": float(avg_losses["fm_loss"]),
            "alpha_ntc": float(alpha_ntc),
            "alpha_updated": bool(alpha_updated),
        })

        if local_rank == 0:
            pbar.set_postfix(loss=cur, lr=f"{cur_lr:.5f}")

        if np.abs(prev_loss - cur) < tol * prev_loss:
            break
        prev_loss = cur

    if is_ddp:
        dist.barrier()
        
        # Broadcast final W to ensure everyone is identical mathematically
        W_final = model.W.detach()
        dist.broadcast(W_final, src=0)
    else:
        W_final = model.W.detach()
    
    # Reconstruct the global H matrix identically across all workers
    with torch.no_grad():
        if is_ddp:
            mask = torch.zeros(k, n, dtype=torch.float32)
            mask[:, local_indices] = 1.0
            # Zero out the columns of H that were entirely processed by other ranks
            H_tensor_cpu *= mask
            # Move to GPU for NCCL all_reduce summation over all distinct local masks
            H_final_gpu = H_tensor_cpu.to(device)
            dist.all_reduce(H_final_gpu, op=dist.ReduceOp.SUM)
            # Now every rank literally possesses the exact FULL H matrix
            H_final_cpu = H_final_gpu.cpu()
        else:
            H_final_cpu = H_tensor_cpu.clone()
    
    cleanup_ddp(is_ddp)
    
    with torch.no_grad():
        W_eff = _act(W_final)
        H_eff = _act(H_final_cpu.to(device))
        if normalize_W:
            s = W_eff.sum(dim=0, keepdim=True).clamp_min(eps)
            W_eff = W_eff / s
            H_eff = H_eff * s.squeeze(0).unsqueeze(1)

    return W_eff.cpu().numpy(), H_eff.cpu().numpy(), pd.DataFrame(loss_history)

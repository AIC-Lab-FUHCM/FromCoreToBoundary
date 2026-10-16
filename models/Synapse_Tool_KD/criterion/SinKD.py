import torch
import torch.nn.functional as F

# def sinkhorn_loss(t, s, lam=10.0, max_iter=50, p=2):
#     """
#     Sinkhorn蒸馏损失 (L_SD)
#     Args:
#         t: Teacher输出 [B, D]
#         s: Student输出 [B, D]
#         lam: 平滑系数 λ (越大 -> 越接近OT距离)
#         max_iter: Sinkhorn归一化迭代次数
#         p: 距离范数类型 (默认L2)
#     Returns:
#         L_SD: Sinkhorn蒸馏损失 (标量)
#     """
#     if t.dim() > 2:
#         t = t.flatten(1)  # [B, D]，把除了batch维外的全部展开
#         s = s.flatten(1)
#     B, D = t.shape

#     # === 1. 构建距离矩阵 D_ij ===
#     # [B, D] → [B, 1, D] 和 [1, B, D]
#     t_expand = t.unsqueeze(1)   # [B, 1, D]
#     s_expand = s.unsqueeze(0)   # [1, B, D]
#     # 距离矩阵 D_{ij} = ||t_i - s_j||_p
#     D_mat = torch.norm(t_expand - s_expand, p=p, dim=2)  # [B, B]

#     # === 2. 构建初始核矩阵 K = exp(-λ * D) ===
#     K = torch.exp(-lam * D_mat)
#     K = K / (K.sum() + 1e-8)  # 防止数值爆炸

#     # === 3. Sinkhorn迭代: 行列归一化 ===
#     for _ in range(max_iter):
#         K = K / (K.sum(dim=1, keepdim=True) + 1e-8)  # Normalize rows
#         K = K / (K.sum(dim=0, keepdim=True) + 1e-8)  # Normalize cols

#     # === 4. 计算Sinkhorn损失 L_SD = <K, D> ===
#     L_SD = torch.sum(K * D_mat)

#     return L_SD

def sinkhorn_loss(t, s, lam=10.0, max_iter=5, p=2, eps=1e-8, normalize=True):
    """Differentiable Sinkhorn distance loss (L_SD).

    Detaches only the teacher tensor ``t`` so gradients flow into the student
    tensor ``s``. L2-normalizes flattened sample vectors and shifts rows by
    their minimum before exponentiation to prevent underflow in ``exp(-lam * D)``.
    """
    if t.dim() == 4 and s.dim() == 4 and t.shape[-2:] != s.shape[-2:]:
        t = F.interpolate(t.detach().float(), size=s.shape[-2:], mode="bilinear", align_corners=False)
    else:
        t = t.detach().float()
    s = s.float()

    if t.dim() > 2:
        t = t.flatten(1)
    if s.dim() > 2:
        s = s.flatten(1)

    if normalize:
        t = F.normalize(t, p=2, dim=1, eps=eps)
        s = F.normalize(s, p=2, dim=1, eps=eps)

    D_mat = torch.cdist(t, s, p=p)
    D_shifted = D_mat - D_mat.detach().min(dim=-1, keepdim=True).values
    K = torch.exp(-lam * D_shifted).clamp_min(eps)
    K = K / (K.sum() + eps)

    for _ in range(max_iter):
        K = K / (K.sum(dim=1, keepdim=True) + eps)
        K = K / (K.sum(dim=0, keepdim=True) + eps)

    K = K / (K.sum() + eps)
    L_SD = torch.sum(K * D_mat)
    return L_SD


def reshape_feat(x):
    c = x.size(1)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, c)
    return x


def pixel_info_nce(pre, pos, neg, temperature=0.1, eps=1e-8):
    """Numerically stable pixel-wise InfoNCE loss in float32."""
    q = F.normalize(reshape_feat(pre.float()), dim=1, eps=eps)
    p = F.normalize(reshape_feat(pos.float()), dim=1, eps=eps)
    n = F.normalize(reshape_feat(neg.float()), dim=1, eps=eps)

    pos_sim = torch.sum(q * p, dim=1) / temperature
    neg_sim = torch.sum(q * n, dim=1) / temperature

    return F.softplus(neg_sim - pos_sim).mean()


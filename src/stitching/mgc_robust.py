"""Robust MGC implementation that handles uniform-edge patches.

Key improvements over basic MGC:
1. Strong regularization on covariance matrix
2. Fallback to SSD when internal gradient variance is too low
3. Normalized scores across directions (consistent magnitude)
"""

import numpy as np


def mgc_robust(img_a, img_b, direction, fallback_threshold=1.0):
    """Compute symmetric MGC with SSD fallback for uniform regions.

    Args:
        img_a, img_b: BGR patches
        direction: 'right' or 'below'
        fallback_threshold: if internal gradient std < this, use SSD instead

    Returns:
        scalar distance (lower = better match), bounded to reasonable range.
    """
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)

    if direction == 'right':
        G_iL = a[:, -1, :] - a[:, -2, :]
        G_ijLR = b[:, 0, :] - a[:, -1, :]
        boundary_a = a[:, -1, :]
        boundary_b = b[:, 0, :]

        G_jR = b[:, 0, :] - b[:, 1, :]
        G_jiRL = a[:, -1, :] - b[:, 0, :]
    elif direction == 'below':
        G_iL = a[-1, :, :] - a[-2, :, :]
        G_ijLR = b[0, :, :] - a[-1, :, :]
        boundary_a = a[-1, :, :]
        boundary_b = b[0, :, :]

        G_jR = b[0, :, :] - b[1, :, :]
        G_jiRL = a[-1, :, :] - b[0, :, :]
    else:
        raise ValueError(direction)

    # Check if internal gradient has enough variation for MGC
    if G_iL.std() < fallback_threshold or G_jR.std() < fallback_threshold:
        # Uniform region — use SSD of boundary pixels
        ssd = np.mean((boundary_a - boundary_b) ** 2)
        return float(ssd)

    # Standard MGC
    mu_a = G_iL.mean(axis=0)
    diff_ia = G_iL - mu_a
    S_a = (diff_ia.T @ diff_ia) / max(len(G_iL) - 1, 1) + np.eye(3) * 1.0  # stronger reg
    try:
        S_a_inv = np.linalg.inv(S_a)
    except np.linalg.LinAlgError:
        return float(np.mean((boundary_a - boundary_b) ** 2))

    diff_a = G_ijLR - mu_a
    d1 = np.mean(np.einsum('ij,jk,ik->i', diff_a, S_a_inv, diff_a))

    mu_b = G_jR.mean(axis=0)
    diff_ib = G_jR - mu_b
    S_b = (diff_ib.T @ diff_ib) / max(len(G_jR) - 1, 1) + np.eye(3) * 1.0
    try:
        S_b_inv = np.linalg.inv(S_b)
    except np.linalg.LinAlgError:
        return float(np.mean((boundary_a - boundary_b) ** 2))

    diff_b = G_jiRL - mu_b
    d2 = np.mean(np.einsum('ij,jk,ik->i', diff_b, S_b_inv, diff_b))

    # Cap extreme values
    return float(min(d1 + d2, 1e6))

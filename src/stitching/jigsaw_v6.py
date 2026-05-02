"""Jigsaw V6: Uses robust MGC that handles uniform edges correctly.

Main change from V4: uses mgc_robust() which falls back to SSD when
internal gradients are degenerate (uniform rows/columns).
"""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from .mgc_robust import mgc_robust


def rotate_patch(img, rot):
    if rot == 0: return img
    k = rot // 90
    return np.rot90(img, k=-k)


class JigsawV6:
    """Jigsaw solver with robust MGC."""

    def __init__(self, grid_size=(15, 15), verbose=True,
                 confidence_thresholds=None, refine_iters=5):
        self.rows, self.cols = grid_size
        self.verbose = verbose
        self.confidence_thresholds = confidence_thresholds or [0.3, 0.5, 0.7, 0.85, 1.0]
        self.refine_iters = refine_iters

    def stitch(self, patches_dict, anchor='patch_0.png'):
        names = list(patches_dict.keys())
        n = len(names)
        ph, pw = patches_dict[names[0]].shape[:2]
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        idx_to_name = {i: nm for nm, i in name_to_idx.items()}

        if self.verbose:
            print(f'[JigsawV6] {n} patches, grid {self.rows}x{self.cols}')
            print('[1/5] Computing robust MGC...')
        compat = self._compute_compat(names, patches_dict)

        if self.verbose:
            print('[2/5] Finding best-buddies...')
        buddies = self._find_best_buddies(compat, n)
        if self.verbose:
            total = sum(len(v) for v in buddies.values())
            print(f'     {total} best-buddy pairs')

        anchor_idx = name_to_idx[anchor]

        if self.verbose:
            print('[3/5] Seeded growth...')
        placement = self._grow(compat, anchor_idx, n)

        if self.verbose:
            print(f'[4/5] Filling remaining ({self.rows*self.cols - len(placement)})...')
        placement = self._fill_remaining(placement, compat, n)

        if self.refine_iters > 0:
            if self.verbose:
                print(f'[5/5] Refinement ({self.refine_iters} iters)...')
            for it in range(self.refine_iters):
                swaps = self._refine_pass(placement, compat)
                if self.verbose:
                    print(f'     Iter {it+1}: {swaps} swaps')
                if swaps == 0:
                    break

        canvas = np.ones((self.rows * ph, self.cols * pw, 3), dtype=np.uint8) * 255
        for (r, c), (idx, rot) in placement.items():
            patch = rotate_patch(patches_dict[idx_to_name[idx]], rot)
            canvas[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = patch

        placement_named = {pos: (idx_to_name[idx], rot) for pos, (idx, rot) in placement.items()}
        return {
            'image': canvas,
            'placement': placement_named,
            'success': len(placement) >= 0.95 * self.rows * self.cols,
            'coverage': len(placement) / (self.rows * self.cols),
            'num_placed': len(placement),
            'num_total': self.rows * self.cols,
        }

    def _compute_compat(self, names, patches_dict):
        n = len(names)
        imgs = [patches_dict[nm] for nm in names]
        compat = {}
        for direction in ['right', 'below']:
            arr = np.full((n, n), np.inf, dtype=np.float32)
            for i in range(n):
                if self.verbose and i % 50 == 0:
                    print(f'     {direction}: {i}/{n}')
                for j in range(n):
                    if i != j:
                        arr[i, j] = mgc_robust(imgs[i], imgs[j], direction)
            compat[direction] = arr
        return compat

    def _find_best_buddies(self, compat, n):
        buddies = {'right': set(), 'below': set()}
        for direction in ['right', 'below']:
            arr = compat[direction]
            best_j = np.argmin(arr, axis=1)
            best_i = np.argmin(arr, axis=0)
            for i in range(n):
                j = best_j[i]
                if best_i[j] == i:
                    buddies[direction].add((i, j))
        return buddies

    def _grow(self, compat, anchor_idx, n):
        placement = {(0, 0): (anchor_idx, 0)}
        placed = {anchor_idx}
        total = self.rows * self.cols

        for threshold in self.confidence_thresholds:
            while len(placement) < total:
                pockets = self._find_pockets(placement)
                if not pockets:
                    break
                best_option = None
                best_rank = (1, 1.0)
                for pos, nbrs in pockets.items():
                    cands = self._rank_cands(nbrs, compat, placed, n)
                    if len(cands) < 2:
                        continue
                    ratio = cands[0][2] / (cands[1][2] + 1e-6)
                    if ratio > threshold:
                        continue
                    rank = (-len(nbrs), ratio)
                    if rank < best_rank:
                        best_rank = rank
                        best_option = (pos, cands[0][0])
                if best_option is None:
                    break
                pos, cand = best_option
                placement[pos] = (cand, 0)
                placed.add(cand)
            if self.verbose:
                print(f'     threshold={threshold:.2f}: placed {len(placement)}/{total}')
            if len(placement) >= total:
                break
        return placement

    def _find_pockets(self, placement):
        pockets = {}
        for (r, c), (idx, rot) in placement.items():
            for dr, dc, direction in [
                (0, 1, 'right_of'), (0, -1, 'left_of'),
                (1, 0, 'below'), (-1, 0, 'above'),
            ]:
                npos = (r + dr, c + dc)
                if not (0 <= npos[0] < self.rows and 0 <= npos[1] < self.cols):
                    continue
                if npos in placement:
                    continue
                pockets.setdefault(npos, []).append((direction, idx, rot))
        return pockets

    def _rank_cands(self, neighbor_info, compat, placed, n):
        scores = []
        for cand in range(n):
            if cand in placed:
                continue
            total = 0.0
            count = 0
            for direction, nbr_idx, _ in neighbor_info:
                if direction == 'right_of':
                    d = compat['right'][nbr_idx, cand]
                elif direction == 'left_of':
                    d = compat['right'][cand, nbr_idx]
                elif direction == 'below':
                    d = compat['below'][nbr_idx, cand]
                elif direction == 'above':
                    d = compat['below'][cand, nbr_idx]
                total += d
                count += 1
            scores.append((cand, 0, total / max(count, 1)))
        scores.sort(key=lambda x: x[2])
        return scores

    def _fill_remaining(self, placement, compat, n):
        placed = {idx for idx, _ in placement.values()}
        empty = [(r, c) for r in range(self.rows) for c in range(self.cols)
                 if (r, c) not in placement]
        unplaced = [i for i in range(n) if i not in placed]
        if not empty or not unplaced:
            return placement

        cost = np.zeros((len(unplaced), len(empty)))
        for ui, patch_idx in enumerate(unplaced):
            for pi, pos in enumerate(empty):
                r, c = pos
                total = 0.0
                count = 0
                for dr, dc, direction in [
                    (0, 1, 'right_of'), (0, -1, 'left_of'),
                    (1, 0, 'below'), (-1, 0, 'above'),
                ]:
                    nbr = (r + dr, c + dc)
                    if nbr in placement:
                        nbr_idx, _ = placement[nbr]
                        if direction == 'right_of':
                            d = compat['right'][nbr_idx, patch_idx]
                        elif direction == 'left_of':
                            d = compat['right'][patch_idx, nbr_idx]
                        elif direction == 'below':
                            d = compat['below'][nbr_idx, patch_idx]
                        elif direction == 'above':
                            d = compat['below'][patch_idx, nbr_idx]
                        total += d
                        count += 1
                cost[ui, pi] = total / max(count, 1) if count > 0 else 1e9

        row_ind, col_ind = linear_sum_assignment(cost)
        for ui, pi in zip(row_ind, col_ind):
            placement[empty[pi]] = (unplaced[ui], 0)
        return placement

    def _refine_pass(self, placement, compat):
        def energy_at(pos):
            if pos not in placement:
                return 0
            r, c = pos
            idx, _ = placement[pos]
            total = 0
            count = 0
            for dr, dc, direction in [
                (0, 1, 'right_of'), (0, -1, 'left_of'),
                (1, 0, 'below'), (-1, 0, 'above'),
            ]:
                nbr = (r + dr, c + dc)
                if nbr in placement:
                    nbr_idx, _ = placement[nbr]
                    if direction == 'right_of':
                        d = compat['right'][idx, nbr_idx]
                    elif direction == 'left_of':
                        d = compat['right'][nbr_idx, idx]
                    elif direction == 'below':
                        d = compat['below'][idx, nbr_idx]
                    elif direction == 'above':
                        d = compat['below'][nbr_idx, idx]
                    total += d
                    count += 1
            return total / max(count, 1)

        energies = {pos: energy_at(pos) for pos in placement}
        positions = sorted(placement.keys(), key=lambda p: -energies[p])

        swaps = 0
        for pos_a in positions[:100]:
            if pos_a == (0, 0):
                continue
            idx_a, rot_a = placement[pos_a]
            best_gain = 0.5
            best_swap = None
            for pos_b in placement:
                if pos_b == (0, 0) or pos_b == pos_a:
                    continue
                idx_b, rot_b = placement[pos_b]
                e_before = energies[pos_a] + energies[pos_b]
                placement[pos_a] = (idx_b, rot_b)
                placement[pos_b] = (idx_a, rot_a)
                e_after = energy_at(pos_a) + energy_at(pos_b)
                placement[pos_a] = (idx_a, rot_a)
                placement[pos_b] = (idx_b, rot_b)
                gain = e_before - e_after
                if gain > best_gain:
                    best_gain = gain
                    best_swap = pos_b
            if best_swap:
                idx_b, rot_b = placement[best_swap]
                placement[pos_a] = (idx_b, rot_b)
                placement[best_swap] = (idx_a, rot_a)
                energies[pos_a] = energy_at(pos_a)
                energies[best_swap] = energy_at(best_swap)
                for base in (pos_a, best_swap):
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nbr = (base[0]+dr, base[1]+dc)
                        if nbr in placement and nbr not in (pos_a, best_swap):
                            energies[nbr] = energy_at(nbr)
                swaps += 1
        return swaps

"""Jigsaw V7: Best-buddy seeded growth with cluster merging.

Targets failure mode: sparse/water-heavy maps where anchor may be in
low-confidence region.

Strategy:
1. Compute robust MGC
2. Find best-buddies and rank them by confidence (ratio of best/2nd)
3. Seed from TOP-K most confident best-buddy pairs as cluster roots
4. Grow each cluster with strict confidence threshold
5. Try to merge clusters with anchor cluster by finding consistent offsets
6. If anchor-growth fails, place anchor cluster by its best inter-cluster edge
7. Fill remaining positions with Hungarian assignment
8. Refinement passes
"""

import numpy as np
import cv2
import time
from scipy.optimize import linear_sum_assignment
from .mgc_robust import mgc_robust
from .jigsaw_v6 import JigsawV6


def rotate_patch(img, rot):
    if rot == 0: return img
    return np.rot90(img, k=-(rot // 90))


class JigsawV7BB:
    """Best-buddy seeded jigsaw solver."""

    def __init__(self, grid_size=(15, 15), verbose=True,
                 confidence_thresholds=None, refine_iters=5,
                 n_seed_clusters=10):
        self.rows, self.cols = grid_size
        self.verbose = verbose
        self.confidence_thresholds = confidence_thresholds or [0.3, 0.5, 0.7, 0.85, 1.0]
        self.refine_iters = refine_iters
        self.n_seed_clusters = n_seed_clusters

    def stitch(self, patches_dict, anchor='patch_0.png'):
        start = time.time()
        names = list(patches_dict.keys())
        n = len(names)
        ph, pw = patches_dict[names[0]].shape[:2]
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        idx_to_name = {i: nm for nm, i in name_to_idx.items()}
        anchor_idx = name_to_idx[anchor]

        if self.verbose:
            print(f'[JigsawV7-BB] {n} patches')

        # Step 1: MGC
        if self.verbose:
            print('[1/5] Computing robust MGC...')
        v6 = JigsawV6(grid_size=(self.rows, self.cols), verbose=False)
        compat = v6._compute_compat(names, patches_dict)

        # Step 2: Best-buddies with confidence
        if self.verbose:
            print('[2/5] Finding best-buddies...')
        bb_pairs = self._find_bb_with_confidence(compat, n)
        if self.verbose:
            print(f'     {len(bb_pairs)} best-buddy pairs')

        # Step 3: Build initial placement by V6 approach from anchor
        if self.verbose:
            print('[3/5] Growing from anchor (V6-style)...')
        placement_anchor = v6._grow(compat, anchor_idx, n)
        if self.verbose:
            print(f'     Anchor cluster: {len(placement_anchor)} patches')

        # Step 4: If anchor growth stalled, augment with best-buddy clusters
        coverage = len(placement_anchor) / (self.rows * self.cols)
        if coverage < 0.95:
            if self.verbose:
                print(f'[4/5] Anchor coverage low ({coverage*100:.0f}%), augmenting with BB clusters')
            placement_anchor = self._grow_with_bb_seeds(
                compat, bb_pairs, placement_anchor, anchor_idx, n
            )
        else:
            if self.verbose:
                print('[4/5] Anchor coverage good, skipping BB augmentation')

        # Fill remaining via Hungarian
        if self.verbose:
            print(f'     Filling remaining: {self.rows*self.cols - len(placement_anchor)}')
        placement = v6._fill_remaining(placement_anchor, compat, n)

        # Step 5: Refinement
        if self.refine_iters > 0:
            if self.verbose:
                print(f'[5/5] Refinement')
            for it in range(self.refine_iters):
                swaps = v6._refine_pass(placement, compat)
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
            'time_sec': time.time() - start,
        }

    def _find_bb_with_confidence(self, compat, n):
        """Find best-buddies with confidence score.
        Returns list of (i, j, direction, confidence) sorted by confidence desc.
        """
        pairs = []
        for direction in ['right', 'below']:
            arr = compat[direction]
            sorted_by_row = np.argsort(arr, axis=1)
            sorted_by_col = np.argsort(arr, axis=0)

            for i in range(n):
                j = sorted_by_row[i, 0]
                if j == i:
                    continue
                # Check j's best i
                best_i_for_j = sorted_by_col[0, j]
                if best_i_for_j != i:
                    continue
                # Both agree — this is a best-buddy
                # Confidence = 1 - (best / second_best)
                best_s = arr[i, sorted_by_row[i, 0]]
                second_s = arr[i, sorted_by_row[i, 1]]
                conf = 1.0 - (best_s / (second_s + 1e-6))
                pairs.append((i, j, direction, conf))

        pairs.sort(key=lambda x: -x[3])
        return pairs

    def _grow_with_bb_seeds(self, compat, bb_pairs, anchor_placement, anchor_idx, n):
        """Try to grow additional sub-clusters from best-buddies, then merge with anchor."""
        total = self.rows * self.cols
        placed = {idx for idx, _ in anchor_placement.values()}
        v6 = JigsawV6(grid_size=(self.rows, self.cols), verbose=False)

        # Try BB pairs as alternative starting points
        # For each BB pair, see if growing from it is viable
        # If we can grow a big cluster that's consistent with anchor's edge, merge

        # Simpler: just try growing from the most-confident patches (not only anchor)
        # Use 'right' BB pairs: they give us two adjacent patches
        candidates_tried = 0
        for i, j, direction, conf in bb_pairs[:self.n_seed_clusters]:
            if i in placed or j in placed:
                continue
            if conf < 0.5:
                break

            # Try placing i-j somewhere compatible with anchor cluster boundary
            # For simplicity: look at anchor's boundary empty pockets and try fitting BB pair
            placed_extended = self._try_fit_bb_pair(
                anchor_placement, compat, i, j, direction, placed, n
            )
            if placed_extended:
                # Now grow from the extended placement using V6 approach
                placement_ext = v6._grow(compat, anchor_idx, n)
                # Actually easier: just keep this extension and try normal growth
                # Re-run anchor growth with the new seed already placed
                placed |= {i, j}
                candidates_tried += 1

        if candidates_tried > 0 and self.verbose:
            print(f'     Added {candidates_tried} BB seeds')

        return anchor_placement

    def _try_fit_bb_pair(self, placement, compat, i, j, direction, placed, n):
        """Try to place best-buddy pair (i, j) with direction 'right' or 'below'
        at a position adjacent to the anchor cluster with high compatibility."""
        # For direction='right': j is right of i
        # For direction='below': j is below i

        # Find empty pockets adjacent to placement with multiple neighbors
        # Try fitting i at a pocket, j at the adjacent position after i
        pockets = {}
        for (r, c), (idx, rot) in placement.items():
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                npos = (r + dr, c + dc)
                if (0 <= npos[0] < self.rows and 0 <= npos[1] < self.cols
                        and npos not in placement):
                    pockets.setdefault(npos, 0)
                    pockets[npos] += 1  # count neighbors

        # Try each pocket
        best_score = float('inf')
        best_placement = None
        for pocket_i, nbrs in pockets.items():
            # Place i at pocket_i, then j at pocket_i + direction offset
            if direction == 'right':
                pocket_j = (pocket_i[0], pocket_i[1] + 1)
            else:
                pocket_j = (pocket_i[0] + 1, pocket_i[1])

            if not (0 <= pocket_j[0] < self.rows and 0 <= pocket_j[1] < self.cols):
                continue
            if pocket_j in placement:
                continue

            # Compute score for i at pocket_i and j at pocket_j
            score = 0
            count = 0
            # i's energy with anchor cluster neighbors
            for dr, dc, dir_ in [
                (0, -1, 'left'),  # left of i = existing
                (-1, 0, 'above'),  # above i
            ]:
                nbr_pos = (pocket_i[0] + dr, pocket_i[1] + dc)
                if nbr_pos in placement:
                    nbr_idx, _ = placement[nbr_pos]
                    if dir_ == 'left':
                        d = compat['right'][nbr_idx, i]
                    else:
                        d = compat['below'][nbr_idx, i]
                    score += d
                    count += 1
            # j's energy with anchor neighbors
            for dr, dc, dir_ in [
                (0, 1, 'right'),  # right of j
                (1, 0, 'below2'),
            ]:
                nbr_pos = (pocket_j[0] + dr, pocket_j[1] + dc)
                if nbr_pos in placement:
                    nbr_idx, _ = placement[nbr_pos]
                    if dir_ == 'right':
                        d = compat['right'][j, nbr_idx]
                    else:
                        d = compat['below'][j, nbr_idx]
                    score += d
                    count += 1

            if count >= 1:
                avg = score / count
                if avg < best_score:
                    best_score = avg
                    best_placement = (pocket_i, pocket_j)

        if best_placement:
            pi, pj = best_placement
            placement[pi] = (i, 0)
            placement[pj] = (j, 0)
            return True
        return False

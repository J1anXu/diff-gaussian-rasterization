import torch
import numpy as np
from tqdm import tqdm


# ============================================
#               KD-tree Block Node
# ============================================
class Block:
    def __init__(self, indices, mins, maxs):
        self.indices = indices                  # numpy array of int
        self.mins    = mins.clone()             # torch [3]
        self.maxs    = maxs.clone()             # torch [3]


# ============================================
#                Split Block
# ============================================
def split_block(block, xyz, max_size):
    """
    xyz: torch [N,3]
    mins/maxs: torch [3]
    """
    pts = xyz[block.indices]  # [M,3] torch

    # 1. 最长维度
    extent = block.maxs - block.mins            # torch [3]
    axis = torch.argmax(extent).item()          # python int

    # 2. 中间值（tensor）
    mid = (block.maxs[axis] + block.mins[axis]) / 2

    # 3. 复制 bounding box (tensor clone)
    left_mins  = block.mins.clone()
    left_maxs  = block.maxs.clone()
    right_mins = block.mins.clone()
    right_maxs = block.maxs.clone()

    # 4. 更新切分位置
    left_maxs[axis]  = mid
    right_mins[axis] = mid

    # 5. 划分 index（tensor compare）
    left_mask  = pts[:, axis] <= mid
    right_mask = pts[:, axis] >  mid

    left_idx  = block.indices[left_mask.cpu().numpy()]
    right_idx = block.indices[right_mask.cpu().numpy()]

    # 6. 生成两个 block
    left_block  = Block(left_idx, left_mins, left_maxs)
    right_block = Block(right_idx, right_mins, right_maxs)

    return left_block, right_block


# ============================================
#               主 KD-tree 构建
# ============================================
def generate_block_masks(xyz, max_size=100000):
    """
    xyz: torch [N,3] (GPU or CPU 都行)
    """
    N = xyz.shape[0]

    mins = xyz.min(dim=0).values   # torch [3]
    maxs = xyz.max(dim=0).values   # torch [3]

    blocks = [Block(np.arange(N), mins, maxs)]
    expected = max(1, N // max_size)

    with tqdm(total=expected, desc="KD-tree splitting") as pbar:
        i = 0
        while i < len(blocks):
            blk = blocks[i]
            if len(blk.indices) > max_size:
                left, right = split_block(blk, xyz, max_size)
                blocks.pop(i)
                blocks.append(left)
                blocks.append(right)
                pbar.update(1)
            else:
                i += 1

    print(f"[INFO] Total blocks: {len(blocks)}")

    block_masks = [blk.indices for blk in blocks]
    for i, mask in enumerate(block_masks):
        print(f"  Block {i:3d}: {len(mask):7d} points")

    return block_masks, blocks

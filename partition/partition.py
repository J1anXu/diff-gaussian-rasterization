import numpy as np
from plyfile import PlyData
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================
#          KD-tree style Split Node
# ============================================
class Block:
    def __init__(self, indices, mins, maxs):
        self.indices = indices    # numpy array
        self.mins = mins          # (3,)
        self.maxs = maxs          # (3,)


def split_block(block, xyz, max_size):
    """
    将 block 继续拆分为两个更小的 block
    （沿最长轴二分）
    """
    pts = xyz[block.indices]

    # 找到当前 block 最大跨度的维度
    extent = block.maxs - block.mins
    axis = np.argmax(extent)      # 0=x, 1=y, 2=z

    # 取该 axis 上的中位数作为切分点
    mid = np.median(pts[:, axis])

    # 左右子块 bounding box
    left_max = block.maxs.copy()
    left_max[axis] = mid

    right_min = block.mins.copy()
    right_min[axis] = mid

    # 切分点
    left_mask = pts[:, axis] <= mid
    right_mask = pts[:, axis] > mid

    left_idx = block.indices[left_mask]
    right_idx = block.indices[right_mask]

    # 生成两个 block
    left_block = Block(left_idx, block.mins.copy(), left_max)
    right_block = Block(right_idx, right_min, block.maxs.copy())

    return left_block, right_block


def generate_block_masks(xyz, max_size=100000):
    N = xyz.shape[0]

    mins = xyz.min(0)
    maxs = xyz.max(0)

    # 初始 block
    blocks = [Block(np.arange(N), mins, maxs)]

    # 预估分裂次数
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

    # 转成 block_masks
    block_masks = [blk.indices for blk in blocks]

    # 输出每个 block 大小
    for i, mask in enumerate(block_masks):
        print(f"  Block {i:3d}: {len(mask):7d} points")
    return block_masks, blocks


# ============================================
#       主函数：均衡 KD-tree 划分
# ============================================
def generate_block_masks_from_ply(ply_path, max_size=100000, vis_path="blocks_visualization.png"):

    print(f"[INFO] Loading {ply_path}")
    ply = PlyData.read(ply_path)

    xyz = np.stack([
        ply['vertex']['x'],
        ply['vertex']['y'],
        ply['vertex']['z']
    ], axis=1)
    N = xyz.shape[0]
    print(f"[INFO] Loaded {N:,} points")
    block_masks, blocks = generate_block_masks(xyz, max_size)

    # 可视化
    visualize_blocks_3d(blocks, vis_path)

    return xyz, block_masks


# ============================================
#           3D 立方体边界可视化
# ============================================
def visualize_blocks_3d(blocks, save_path):

    print(f"[INFO] Rendering 3D block visualization → {save_path}")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab20(np.linspace(0, 1, len(blocks)))

    for i, blk in enumerate(blocks):
        draw_cube(ax, blk.mins, blk.maxs, colors[i % 20], alpha=0.25)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Balanced KD-Tree Blocks (3D Bounding Boxes)")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("[INFO] Saved.")


# ============================================
#       绘制一个立方体（bounding box）
# ============================================
def draw_cube(ax, mins, maxs, color, alpha=0.25):
    x1, y1, z1 = mins
    x2, y2, z2 = maxs

    # 立方体 8 个顶点
    pts = np.array([
        [x1, y1, z1],
        [x2, y1, z1],
        [x2, y2, z1],
        [x1, y2, z1],
        [x1, y1, z2],
        [x2, y1, z2],
        [x2, y2, z2],
        [x1, y2, z2]
    ])

    # 每个面的顶点索引
    faces = [
        [pts[0], pts[1], pts[2], pts[3]],
        [pts[4], pts[5], pts[6], pts[7]],
        [pts[0], pts[1], pts[5], pts[4]],
        [pts[2], pts[3], pts[7], pts[6]],
        [pts[1], pts[2], pts[6], pts[5]],
        [pts[4], pts[7], pts[3], pts[0]],
    ]

    ax.add_collection3d(Poly3DCollection(
        faces, facecolors=color, linewidths=0.5, edgecolors="k", alpha=alpha
    ))



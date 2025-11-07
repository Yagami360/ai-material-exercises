import os
import glob

work_dir = "../"
results_dir = os.path.join(work_dir, "results")

# DDPによって作成されたファイルは、通常、複数の結果ファイルが存在するため、globを使用します。
# num_gpus=1 の場合、一つのファイルのみが存在します。
import glob
results_files = glob.glob(os.path.join(results_dir, "*.npy"))
print("Generated files:")
for f in results_files:
    print(f" - {f}")

# 出力例:
# Generated files:
#  - /workspace/bionemo2/results/embeddings_rank_0.npy
#  - /workspace/bionemo2/results/hiddens_rank_0.npy
#  - /workspace/bionemo2/results/logits_rank_0.npy


# ------------------------------------------------------------
# 結果ファイルをNumPy配列としてロードし、その形状を確認できます。
# ------------------------------------------------------------
import numpy as np

# 埋め込み
embeddings = np.load(os.path.join(results_dir, "embeddings_rank_0.npy"))
print(f"Embeddings shape: {embeddings.shape}")

# 隠れ状態
hiddens = np.load(os.path.join(results_dir, "hiddens_rank_0.npy"))
print(f"Hiddens shape: {hiddens.shape}")

# ロジット
logits = np.load(os.path.join(results_dir, "logits_rank_0.npy"))
print(f"Logits shape: {logits.shape}")

# 出力例:
# Embeddings shape: (10, 1280)
# Hiddens shape: (10, 62, 1280)
# Logits shape: (10, 62, 33)

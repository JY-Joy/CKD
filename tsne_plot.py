import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PCA_components = 50
postfix = "before"
num_groups = 1000
group_size = 64

# learned_embs = np.load(f"../img_embs_{postfix}.npy")
# gt_embs = np.load("../img_embs_gt.npy")

# pca = PCA(n_components=PCA_components)
# pca_x = pca.fit_transform(learned_embs.reshape(64000, -1))
# pca_labels = pca.fit_transform(gt_embs.reshape(1000, -1))
# pca_components_x = pca_x.reshape(1000, 64, -1)
# pca_components_labels = pca_labels.reshape(1000, 1, -1)

# tsne = TSNE(n_components=2, random_state=42)
# tsne_label = tsne.fit_transform(pca_labels)
# np.save(f"../tsne_label.npy", tsne_label)
# pca_components_concat = np.concatenate([pca_components_labels, pca_components_x], axis=1).reshape(65000, -1)
# tsne_concat = tsne.fit_transform(pca_components_concat)
# np.save(f"../tsne_labelCat_{postfix}.npy", tsne_concat)

cmap = plt.get_cmap('viridis')
colors = [cmap(i/num_groups) for i in range(num_groups)]
plt.figure(figsize=(10, 8))

# tsne_concat = np.load(f"../tsne_labelCat_{postfix}.npy")
# tsne_label = np.load("../tsne_label.npy")
# tsne_x = tsne_concat.reshape(1000, 65, -1)[:,1:65,:]
# tsne_label = tsne_concat.reshape(1000, 65, -1)[:,0,:]
# print(tsne_x.shape, tsne_label.shape)
# for i in range(num_groups):
#     group_data = tsne_x[i]
#     plt.scatter(group_data[:, 0], group_data[:, 1], color=colors[i], alpha=0.01, s=100, edgecolors='none')
# for i in range(num_groups):
#     plt.scatter(tsne_label[i, 0], tsne_label[i, 1], color=colors[i], marker="*", alpha=1, s=200)

tsne_x = np.load(f"../tsne_{postfix}.npy").reshape(1000, 64, -1)
tsne_concat = np.load(f"../tsne_concat_{postfix}.npy")
# tsne_x = tsne_concat.reshape(1000, 65, -1)[:,1:65,:]
tsne_labels = tsne_concat.reshape(1000, 65, -1)[:,0,:]
for i in range(num_groups):
    group_data = tsne_x[i]
    plt.scatter(group_data[:, 0], group_data[:, 1], color=colors[i], alpha=1/64, s=150, edgecolors='none')
for i in range(num_groups):
    plt.scatter(tsne_labels[i, 0], tsne_labels[i, 1], color=colors[i], marker="*", alpha=1, s=150)

plt.title('t-SNE')
plt.savefig('../tsne.png')

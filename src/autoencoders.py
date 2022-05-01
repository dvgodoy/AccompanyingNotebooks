import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from torchvision.transforms import ToTensor
from PIL import Image

def show(tensor, ax=None):
    img = np.rollaxis(tensor.detach().cpu().numpy(), 0, 3)
    kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1} if img.shape[-1] == 1 else {}
    if ax is None:
        plt.imshow(img, **kwargs)
    else:
        ax.imshow(img, **kwargs)

def set_fontsize(ax, size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)

class AEViz(object):
    def __init__(self, enc, dec, data, z_size=None):
        self.enc = enc
        self.dec = dec
        if z_size is None:
            z_size = enc.z_size
        self.z_size = z_size
        self.data = data
        self.device = next(enc.parameters()).device.type
        self.input_size = next(iter(data))[0].shape[1:]
        self.in_channels = self.input_size[0] if len(self.input_size) > 2 else 1
        self.img_size = self.input_size[-2:] if len(self.input_size) > 2 else self.input_size

    def plot_latent(self, num_batches=100, figsize=(8, 8), ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        for i, (x, y) in enumerate(self.data):
            if i <= num_batches:
                z = self.enc(x.to(self.device))
                z = z.detach().cpu().numpy().squeeze()
                if z.ndim > 1:
                    sp = ax.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
                else:
                    ax.scatter(z, y, color='C0', alpha=0.5)
        if z.ndim == 1:
            ax.set_xlabel('Z')
            ax.set_ylabel('Radius')
        if z.ndim > 1:
            fig.colorbar(sp)
            
        fig.tight_layout()
        return fig

    def plot_reconstructed(self, r0=(-3, 3), r1=(-3, 3), n=12, figsize=(8, 8), ax=None):
        if self.z_size <=2 :
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
            else:
                fig = ax.get_figure()

            lin0 = np.linspace(*r0, n)
            lin1 = np.linspace(*r1, n) if self.z_size > 1 else [0]
            ny = n if self.z_size > 1 else 1
            img = np.zeros((self.in_channels, ny*self.img_size[0], n*self.img_size[1]))
            if self.z_size > 1:
                z = torch.as_tensor(np.concatenate([g.reshape(-1, 1) for g in np.meshgrid(lin0, lin1)], axis=1)).float()
            else:
                z = torch.as_tensor([lin0]).float().reshape(-1, 1)
            x_hat = self.dec(z.to(self.device)).detach().cpu().numpy()
            x_hat = np.rollaxis(x_hat, 1, 4)
            img = np.concatenate([np.concatenate(r, axis=1) for r in (np.split(x_hat, n, 0) if self.z_size > 1 else x_hat)], axis=int(self.z_size==1))
            kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1} if self.in_channels == 1 else {}
            ax.imshow(img, **kwargs)
            if self.z_size == 1:
                ax.set_yticks([])
                ax.set_xticks(np.linspace(0.5, 0.5+n-1, n)*img.shape[0])
                ax.set_xticklabels([f'{v:.2f}' for v in lin0])
                ax.set_xlabel('Z')
            fig.tight_layout()
            return fig
        else:
            return self._plot_umap_reconstruced()
            

    def plot_z_dist(self, num_batches=100, n_cols=3, bins=100, axs=None):
        zs = []
        for x, y in self.data:
            z = self.enc(x.to(self.device))
            zs.append(z.detach().cpu().numpy())
        zs = np.concatenate(zs, axis=0)

        n_cols = min(self.z_size, n_cols)
        n_rows = self.z_size // n_cols + int(self.z_size % n_cols > 0)
        if axs is None:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))

        axs = np.atleast_2d(axs).flatten()
        fig = axs[0].get_figure()
        x = np.linspace(-3, 3, 101)
        for zdim in range(self.z_size):
            axs[zdim].hist(zs[:, zdim], density=True, bins=bins)
            axs[zdim].plot(x, norm.pdf(x))
            axs[zdim].set_title(f'Dim {zdim}')
            axs[zdim].set_xlabel('Z')
        for zdim in range(self.z_size, n_cols*n_rows):
            axs[zdim].axis('off')
        fig.tight_layout()
        return fig

    def _plot_umap_reconstruced(num_batches=100):
        # encode data into latent space Z
        zs = []
        labels = []
        for x, y in self.data:
            z = self.enc(x.to(self.device))
            zs.append(z.detach().cpu().numpy())
            labels.append(y.cpu().numpy())
        zs = np.concatenate(zs, axis=0)
        labels = np.concatenate(labels, axis=0)

        # use UMAP to map n-dim Z into 2D Z
        mapper = umap.UMAP(random_state=42).fit(zs)
        coords = mapper.transform(zs)

        # builds grid in 2D
        maxs = coords.max(axis=0)
        mins = coords.min(axis=0)    
        corners = np.array([
            [mins[0], mins[1]],  # 1
            [mins[0], maxs[1]],  # 7
            [maxs[0], mins[1]],  # 2
            [maxs[0], maxs[1]],  # 0
        ])
        grid_pts = np.array([
            (corners[0]*(1-x) + corners[1]*x)*(1-y) +
            (corners[2]*(1-x) + corners[3]*x)*y
            for y in np.linspace(0, 1, 10)
            for x in np.linspace(0, 1, 10)
        ])  

        # maps grid back into n-dim Z
        inv_transformed_points = torch.as_tensor(mapper.inverse_transform(grid_pts)).to(self.device)
        x_hats = self.dec(inv_transformed_points).detach().cpu().numpy()

        # Set up the grid
        fig = plt.figure(figsize=(12,6))
        gs = GridSpec(10, 20, fig)
        scatter_ax = fig.add_subplot(gs[:, :10])
        digit_axes = np.zeros((10, 10), dtype=object)
        for i in range(10):
            for j in range(10):
                digit_axes[i, j] = fig.add_subplot(gs[i, 10 + j])

        # Use umap.plot to plot to the major axis
        # umap.plot.points(mapper, labels=labels, ax=scatter_ax)
        scatter_ax.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1],
                           c=labels.astype(np.int32), cmap='Spectral', s=15)
        scatter_ax.set(xticks=[], yticks=[])
        # Plot the locations of the text points
        scatter_ax.scatter(grid_pts[:, 0], grid_pts[:, 1], marker='x', c='k', s=15)

        for i in range(10):
            for j in range(10):
                x_hat = np.rollaxis(x_hats[i*10+j], 0, 3)
                digit_axes[i, j].imshow(x_hat)
                digit_axes[i, j].set(xticks=[], yticks=[])

        return fig
    
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def figure1(dataset):
    real = dataset.tensors[0][:10].numpy()
    real = np.rollaxis(real, 1, 4)
    img_real = np.concatenate([np.concatenate(r, axis=1) for r in np.split(real, 2, 0)], axis=0)

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    axs = axs.flatten()
    for i in range(10):
        axs[i].set_title(f'Image #{i}')
        axs[i].imshow(real[i], cmap='gray', vmin=0, vmax=1)
    fig.tight_layout()
    return fig

def figure2(autoencoder, image, device):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    show(image, ax=axs[0])
    z = autoencoder.enc(image.to(device))
    show(autoencoder.dec(z)[0], ax=axs[2])
    axs[0].set_title('Original')
    axs[1].axis('off')
    axs[1].annotate(f'z = [{z.item():.4f}]', (0.25, .5), fontsize=20)
    axs[1].set_title('Latent Space')
    axs[2].set_title('Reconstructed')
    for i in range(3):
        set_fontsize(axs[i], 20)
    return fig

def figure4(autoencoder, device):
    fig, axs = plt.subplots(1, 5, figsize=(20, 6))
    for i, z in enumerate([-3., -.5, 0.0, .9, 3]):
        show(autoencoder.dec(torch.tensor([[z]]).float().to(device))[0], ax=axs[i])
        axs[i].set_title(f'z = [{z:.4f}]')
        axs[i].axis('off')
    for i in range(5):
        set_fontsize(axs[i], 20)
    fig.tight_layout()
    return fig

def figure5(autoencoder, image_fname, device):
    img = ToTensor()(Image.open(image_fname))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    show(img, ax=axs[0])
    z = autoencoder.enc(img.to(device))
    show(autoencoder.dec(z)[0], ax=axs[2])
    axs[0].set_title('Original')
    axs[1].axis('off')
    axs[1].annotate(f'z = [{z.item():.4f}]', (0.25, .5), fontsize=20)
    axs[1].set_title('Latent Space')
    axs[2].set_title('Reconstructed')
    for i in range(3):
        set_fontsize(axs[i], 20)
    return fig

def kl_div(mu, std):
    kl_div = -0.5*(1 + np.log(std**2) - mu**2 - std**2)
    return kl_div

def figure6():
    x = np.linspace(-5, 5, 101)
    normal = norm.pdf(x, loc=0, scale=1)

    nc = 5
    nr = 8

    mus = np.linspace(-3, 3, nc)
    scales = np.linspace(0.2, 3, nr)
    xx, yy = np.meshgrid(mus, scales)

    fig, axs = plt.subplots(nr, nc, figsize=(nc*4, nr*2))
    divergences = np.zeros((nr, nc))

    for row in range(nr):
        for col in range(nc):
            dist = norm.pdf(x, loc=xx[row, col], scale=yy[row, col])
            if row == 0:
                axs[row, col].set_title(f'mu = {xx[row, col]:.2f}')
            if col == 0:
                axs[row, col].set_ylabel(f'std = {yy[row, col]:.2f}')
            divergences[row, col] = kl_div(xx[row, col], yy[row, col])
            axs[row, col].set_xlabel(f'KL div = {divergences[row, col]:.2f}')
            axs[row, col].plot(x, normal)
            axs[row, col].plot(x, dist)
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            set_fontsize(axs[row, col], 20)
    fig.tight_layout()
    return fig, mus, scales, divergences

def figure7(mus, scales, divergences):
    im, cbar = heatmap(divergences, row_labels=[f'std = {s:.2f}' for s in scales], col_labels=[f'mu = {m:.2f}' for m in mus])
    return im

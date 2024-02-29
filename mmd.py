'''Credit: https://github.com/ekorman/neurve'''

import torch

SMALL = 1e-12


def distmfld(q1, c1, q2, c2):
    """ Computes (batchwise) the distance between two batches of points in a
    manifold.

    Parameters
    ----------
    q1 : torch.tensor
        shape [nc, n] giving chart membership probabilities for the first batch
        of points.
    c1 : torch.tensor
        shape [nc, n, d] giving the coordinates for the first batch of points.
    q2 : torch.tensor
        shape [nc, n] giving chart membership probabilities for the second batch
        of points.
    c2 : torch.tensor
        shape [nc, n, d] giving the coordinates for the first batch of points.

    where nc is the number of coordinate charts, n is the number of points
    (e.g. batch size), and d is the dimension of the manifold.

    Returns
    -------
    torch.tensor
        has shape n. the ith component is the manifold distance between the point
        defined by q1[:, i], c1[:, i, :] and the point q2[:, i], c2[:, i, :]
    """
    q1q2_sum = (q1 * q2).sum(0)
    return ((q1 * q2 * ((c1 - c2) ** 2).mean(2)).sum(0) / q1q2_sum).masked_fill(
        q1q2_sum == 0, 1
    )


def pdist_mfld(q1, c1, q2, c2):
    """
    Parameters
    ----------
    q1 : torch.tensor
        shape [nc, m]
    c1 : torch.tensor
        shape [nc, m, d]
    q2 : torch.tensor
        shape [nc, n]
    c2 : torch.tensor
        shape [nc, n, d]

    Returns
    -------
    torch.tensor
        shape [m, n]
    """
    nc, m, d = c1.shape
    _, n = q2.shape
    x1_norm2 = (c1 ** 2).sum(2).unsqueeze(2) @ torch.ones((nc, 1, n), device=c1.device)
    x2_norm2 = torch.ones((nc, m, 1), device=c1.device) @ (c2 ** 2).sum(2).unsqueeze(1)
    x1_dot_x2 = c1 @ c2.transpose(1, 2)

    q1q2_sum = q1.T @ q2

    ret = x1_norm2 - 2 * x1_dot_x2 + x2_norm2
    ret = (ret * q1.unsqueeze(2) * q2.unsqueeze(1)).sum(0)
    ret = ret / (q1q2_sum + SMALL) / d
    ret = ret.masked_fill(q1q2_sum <= SMALL, 1)
    return ret


def batch_pdist(X, Y, device=None):
    """ Computes all the pairwise distances.

    Parameters
    ----------
    X : torch.tensor
        shape [c, n, d]
    Y : torch.tensor
        shape [c, m, d] or [1, m, d]

    Returns
    -------
    torch.tensor
        output has shape [c, n, m]. The entry at i, j, k is the squared distance
        between the d-dimensional vectors X[i, j] and Y[i, k] if Y.shape[0] is c
        and Y[0, k] if Y.shape[0] is 1.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = X.shape[0]
    n, m = X.shape[1], Y.shape[1]
    X_norm2 = (X ** 2).sum(2)
    Y_norm2 = (Y ** 2).sum(2)
    X_dot_Y = torch.matmul(X, Y.transpose(1, 2))

    return (
        torch.matmul(X_norm2.unsqueeze(2), torch.ones((c, 1, m), device=device))
        - 2 * X_dot_Y
        + torch.matmul(torch.ones((c, n, 1), device=device), Y_norm2.unsqueeze(1))
    )


def pdist(X, Y):
    """ Computes all the pairwise distances

    Parameters
    ----------
    X : torch.tensor
        shape [n, d]
    Y : torch.tensor
        shape [m, d]

    Returns
    -------
    torch.tensor
        shape [n, m] of all pairwise distances
    """
    n, m = X.shape[0], Y.shape[0]
    X_norm2 = (X ** 2).sum(1)
    Y_norm2 = (Y ** 2).sum(1)
    X_dot_Y = X @ Y.T

    return (
        X_norm2.unsqueeze(1) @ torch.ones((1, m), device=X.device)
        - 2 * X_dot_Y
        + torch.ones((n, 1), device=Y.device) @ Y_norm2.unsqueeze(0)
    )


def psim(X, Y):
    """ Computes all the pairwise similarities

    Parameters
    ----------
    X : torch.tensor
        shape [n, d]
    Y : torch.tensor
        shape [m, d]

    Returns
    -------
    torch.tensor
        shape [n, m] of all pairwise similarities
    """
    n, m = X.shape[0], Y.shape[0]
    X_norm = ((X ** 2).sum(1) + SMALL).sqrt()
    Y_norm = ((Y ** 2).sum(1) + SMALL).sqrt()
    X_dot_Y = X @ Y.T

    ret = X_dot_Y / (
        (X_norm.unsqueeze(1) @ torch.ones((1, m), device=X.device))
        * (torch.ones((n, 1), device=Y.device) @ Y_norm.unsqueeze(0))
    )

    return ret


def sigmoid_inv(x):
    return torch.clamp(torch.log(x / (1 - x)), -10000, 10000)


def rbf(X, Y, sigma, batch=False):
    d = batch_pdist if batch else pdist
    return torch.exp(-d(X, Y) / 2 / sigma ** 2)


def imq(X, Y, C, batch=False):
    d = batch_pdist if batch else pdist
    return C / (C + d(X, Y))


def q_loss(q):
    """
    Parameters
    ----------
    q : torch.Tensor
        shape [n, nc] giving chart membership probabilities, where
        n is the batch size, nc the number of charts
    """
    return ((q - 1 / q.shape[1]) ** 2).sum(1).mean()


class MMD:
    def __init__(self, kernel, sigma):
        assert kernel in ["rbf", "imq"]
        k = rbf if kernel == "rbf" else imq
        self.k = lambda X, Y: k(X, Y, sigma)

    def __call__(self, X, Y):
        """ Computes the MMD between samples

        x : [N, d]
        """
        n = X.shape[0]
        assert n == Y.shape[0]

        ret = (self.k(X, X).sum() - n) / (n * (n - 1))
        ret += (self.k(Y, Y).sum() - n) / (n * (n - 1))
        ret += -2 * self.k(X, Y).sum() / n ** 2
        return ret


class MMDManifold:
    def __init__(self, kernel, sigma):
        assert kernel in ["rbf", "imq"]
        k = rbf if kernel == "rbf" else imq
        self.k = lambda X, Y: k(X, Y, sigma, batch=True)

    def __call__(self, q, X, Y):
        """ X is the encoder over the data distribution,
        q is the coordinate probabilities for X,
        and Y is sampled from the ambient distribution

        Parameters
        ----------
        q : torch.Tensor
            shape [nc, n] giving chart membership probabilities
        X : torch.Tensor
            shape [nc, n, d] (nc the number of charts, d the ambient dimension)
        Y : torch.Tensor
            shape [n, d] (d the manifold dimension)
        """
        nc, n = X.shape[:2]
        assert n == Y.shape[0]

        Y = Y.unsqueeze(0)

        qq = q.unsqueeze(2) * q.unsqueeze(1)  # shape [nc, N, N]

        ret = (
            self.k(X, X) * qq * (1 - torch.eye(n, device=q.device).unsqueeze(0))
        ).sum() / (n * (n - 1))
        ret += (self.k(Y, Y).sum() - n) / (nc * n * (n - 1))
        ret += -2 * (q.unsqueeze(2) * self.k(X, Y)).sum() / (nc * n ** 2)
        return ret


class MMDManifoldLoss:
    def __init__(self, kernel, sigma, device):
        """
        Parameters
        ----------
        kernel : str
            one of "imq" or "rbf"
        sigma : float
        device : str
        """
        self.mmd = MMDManifold(kernel=kernel, sigma=sigma)
        self.device = device

    def __call__(self, q, X):
        """
        Parameters
        ----------
        q : torch.Tensor
            shape [n, nc] giving chart membership probabilities, where
            n is the batch size, nc the number of charts
        X : torch.Tensor
            coordinates. should be shape [n, nc, d] (d the dimension of the
            manifold). Note that X should be the output before the sigmoid activation
            since the MMD will be computed between the sample X and the inverse
            sigmoid of a random sample of the uniform distribution on [0, 1]^d

        Returns
        -------
        torch.Tensor
            the MMD loss (a scalar)
        """
        return self.mmd(
            q.T,
            X.transpose(0, 1),
            sigmoid_inv(torch.rand(X.shape[0], X.shape[2], device=self.device)),
        )

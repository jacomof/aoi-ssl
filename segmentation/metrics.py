import torch
from torch.linalg import svdvals
from numpy.linalg import eigvals
from enum import Enum
import powerlaw

class metricType(Enum):
    auc = "auc"
    entropy = "entropy"

class Metrics:
    """Class to compute metrics for embeddings.
    
    Args:
        embeddings (torch.Tensor): The embeddings of the input.
        approx (bool): Whether to use the approximate SVD or the exact one. Default is True.
    """

    def __init__(self, embeddings: torch.Tensor, approx: bool = True):
        self.embeddings = embeddings
        print("Embeddings sum: ", embeddings.sum().item())
        self.svs, self.num_svs = self._compute_svd(embeddings, approx=approx, normalize=True)

    def _compute_eigenvalues(self, embs: torch.Tensor):
        """Compute the eigenvalues of the embeddings.
        
        Args:
            embs (torch.Tensor): The embeddings of the input.

        Returns:
            numpy.array: The eigenvalues of the embeddings.
        """

        # If the embeddings are 2D, we flatten them to 1D.
        if len(embs.size()) > 2:
            embs = embs.view(embs.size(0), -1)

        # norm used to transform embs into unit vectors so the SVDs are invariant to scaling. 
        #norms = torch.linalg.norm(embs, dim=1)
        eivals = eigvals(embs.cpu().numpy())
        return eivals

    def _compute_svd(self, embs: torch.Tensor, approx: bool = True, normalize: bool = False):
        """Compute the singular values of the embeddings.

        Args:
            embs (torch.Tensor): The embeddings of the input.
            approx (bool): Whether to use the approximate SVD or the exact one.
            normalize (bool): Whether to normalize the embeddings.
        
        Returns:
            torch.Tensor: The singular values of the embeddings.
        """
        # If the embeddings are 2D, we flatten them to 1D.
        print("Embeddings shape: ", embs.shape)
        if len(embs.size()) > 2:
            embs = embs.view(embs.size(0), -1)

        # If the embeddings are 1D, we add a dimension to them.
        if len(embs.size()) == 1:
            embs = embs.unsqueeze(0)
        
        emb_dim = embs.size(1)
        batch_dim = embs.size(0)

        embs -= torch.mean(embs, dim=0, keepdim=True)

        # norm used to transform embs into unit vectors so the SVDs are invariant to scaling. 
        if normalize:
            norms = torch.linalg.norm(embs, dim=1)

            # We add a small value to avoid division by zero (i.e numerical stability)
            # Transform SVDs into unit vectors. I think they do this to map the AUC 
            # into the range [0.5, 1].
            embs = embs/(1e-6 + norms.unsqueeze(1))
        num_svs = min(batch_dim, emb_dim)
        print("SVD approximation rank: ", num_svs)
        if approx:
            # The rank of the feature matrix won't be larger than the batch or embedding dimension, 
            # so we take the minimum between them and k.
            # Context ensures correct behavior in mixed precision training.
            with torch.autocast(device_type=embs.device.type, enabled=False):
                embs = embs.float()
                _, svs,_ = torch.svd_lowrank(embs, q=num_svs)
        else:
            svs = svdvals(embs.cpu().numpy())
            svs = torch.tensor(svs).to(embs.device)

        return svs, num_svs

    def auc_embedding_collapse(
            self,
            ):
        """Calculate the Area Under the Curve (AUC) for the singular value decomposition of the embeddings.
        
        It's a measure of dimensional collapse based on the paper "Understanding Collapse in Non-Contrastive
        Siamese Representation Learning" by A. C. Li et al. (2022). It should be interpreted
        as a measure of the quality of the embeddings, and always in tandem with the model loss
        (i.e no dimensional colapse doesn't mean that the embeddings are good).

        Args:
            embs (torch.Tensor): The embeddings of the input.
            approx (bool): Whether to use the approximate SVD or the exact one.
        
        Returns:
            torch.Tensor: The AUC of the singular values of the embeddings.
        """

        svd_sum = torch.sum(self.svs)
        svd_max = torch.max(self.svs)
        svd_min = torch.min(self.svs)
        svd_cum_sum = torch.cumsum(self.svs, dim=0)
        auc_scaled = torch.sum(svd_cum_sum)/self.num_svs
        print("svd sum is: ", svd_sum.item())
        print("svd max is: ", svd_max.item())
        print("svd min is: ", svd_min.item())
        #print("svd cumulative sum is: ", svd_cum_sum)
        if svd_sum == 0:
            print("svd sum is 0, returning 0")
            return 0
        auc_norm = auc_scaled / svd_sum
        auc = auc_norm.item()
        assert auc >= 0.49 and auc <=1.1, f"AUC value is {auc}, which is not in the range [0.5, 1]"
        return auc

    def entropy_embedding_collapse(
            self,
        ):
        """
        Calculate the entropy for the singular value decomposition of the embeddings.
        
        It's a measure of dimensional collapse based on the paper: "Assessing the downstream
        performance of pretrained self-supervised representations by their rank" by Garrido et al. (2022).
        It should be interpreted as a measure of the quality of the embeddings, and always in tandem with the model loss
        (i.e no dimensional colapse doesn't mean that the embeddings are good).
        
        Returns:
            torch.Tensor: The entropy of the singular values of the embeddings.
        
        """

        # Normalize the singular values
        # Consider that the logarithm of 0 is undefined! 
        # So add small value for numerical stability
        svs = (self.svs/self.svs.sum()) + 1e-6
        entropy = -torch.sum(svs * torch.log(svs))
        return entropy.item()

    def power_law_embedding_collapse(
            self,
        ):
        """Calculate the power law for the singular value decomposition of the embeddings."
        
        Based on the paper: "Investigating Power laws in Deep Representation Learning"
        by Ghosh et al. (2022).

        Args:
            embs (torch.Tensor): The embeddings of the input.

        Returns:
            float: The power law of the singular values of the embeddings.
        """

        covar = self.embeddings.T @ self.embeddings
        eivals = self._compute_eigenvalues(covar)
        fit = powerlaw.Fit(eivals)
        alpha = fit.alpha
        return alpha


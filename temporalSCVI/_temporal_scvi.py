import logging
from functools import partial
from typing import Optional, Sequence

from anndata import AnnData
import numpy as np
import pandas as pd
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, RNASeqMixin, VAEMixin
from scvi.model._utils import _get_var_names_from_setup_anndata
import torch


from ._random_time import RandomTime

logger = logging.getLogger(__name__)


class TemporalSCVI(RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Skeleton for an scvi-tools model.

    Please use this skeleton to create new models.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    **model_kwargs
        Keyword args for :class:`~mypackage.MyModule`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.data.setup_anndata(adata, batch_key="batch")
    >>> vae = mypackage.MyModel(adata)
    >>> vae.train()
    >>> adata.obsm["X_mymodel"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        **model_kwargs,
    ):
        super(TemporalSCVI, self).__init__(adata)

        # self.summary_stats provides information about anndata dimensions and other tensor info

        self.module = RandomTime(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            **model_kwargs,
        )
        self._model_summary_string = "Overwrite this attribute to get an informative representation for your model"
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @torch.no_grad()
    def get_latent_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        return_std = False
    ) -> np.ndarray:
        r"""
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")
        
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent = []
        latent_std = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            qt_m = outputs["qt_m"]
            qt_v = outputs["qt_v"]
            t = outputs["t"]

            if give_mean:
                t = qt_m

            latent += [t.cpu()]
            latent_std += [torch.sqrt(qt_v).cpu()]

        if return_std:
            return torch.cat(latent).numpy(), torch.cat(latent_std).numpy()

        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_temporal_expression(
        self,
        start: float,
        end: float,
        steps: int = 128,
        n_samples: int = 128,
        adata: Optional[AnnData] = None,
        gene_list: Optional[Sequence[str]] = None
    ):
        r"""
        Generate expression levels over time by integrating out non-temporal variation.
        """
        adata = self._validate_anndata(adata)

        all_genes = _get_var_names_from_setup_anndata(adata)
        if gene_list is None:
            gene_mask = slice(None)
            filtered_gene_list = all_genes
        else:
            gene_mask = [True if gene in gene_list else False for gene in all_genes]
            filtered_gene_list = [gene for gene in gene_list if gene in all_genes]

        idx = np.random.choice(np.arange(adata.shape[0]), n_samples)
        
        # Sample cell representations
        t_Z = torch.from_numpy(self.get_latent_representation(adata=adata, indices=idx, give_mean=False))
        t_Z_input = t_Z.repeat_interleave(steps, 0)
        replicate = torch.arange(n_samples)[:, None].repeat_interleave(steps, 0)

        # Grid over time points
        cont_covs = torch.linspace(start, end, steps)[:, None]
        cont_covs_input = cont_covs.repeat(n_samples, 1)

        library_input = torch.ones((steps, 1)).repeat(n_samples, 1)
        generative_output = self.module.generative(t_Z_input, library_input, cont_covs_input)

        output = generative_output['px_rate']
        output = output[..., gene_mask]
        output = output.detach().cpu().numpy()

        df1 = pd.DataFrame({'time': cont_covs_input.numpy()[:, 0], 'sample': replicate.numpy()[:, 0]})
        df2 = pd.DataFrame(output, columns=filtered_gene_list)
        dfb = pd.concat((df1, df2), axis=1)
        long_dfb = dfb.melt(id_vars=['time', 'sample'], var_name='gene', value_name='px_scale')

        return long_dfb

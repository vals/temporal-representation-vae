from scvi.data import synthetic_iid
from temporalSCVI import TemporalSCVI
import pyro


def test_temporal_scvi():
    n_latent = 5
    adata = synthetic_iid()
    model = TemporalSCVI(adata, n_latent=n_latent)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_elbo()
    model.get_latent_representation()
    model.get_marginal_ll(n_mc_samples=5)
    model.get_reconstruction_error()
    model.history

    # tests __repr__
    print(model)

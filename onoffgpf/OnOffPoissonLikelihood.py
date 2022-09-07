from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from gpflow.base import Parameter
from gpflow.utilities import positive
from gpflow.likelihoods import Poisson
from gpflow.config import default_float

float_type = default_float()
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class OnOffPoissonLikelihood(Poisson):
    """
    Computes\int {log(y|f) [\int p(f,g) dg]} df = \int {log(y|f) [\int p(f|g) p(g) dg]} df
    Where
    p(y) = N(y|A*fmean,sigma2)
    p(f|g) = N(f| diag(gmean)*fmean,diag(gvar)*fvar)
    p(g) = N(gmean,gvar)

    While marginalising gamma, an uncertainity with respect to mean is introduced as a trace term
    (This term is in addition to standard SVGP classification terms)
    """

    def __init__(self,):
        super().__init__()

    # not implemented logp, conditional_mean and others

    def variational_expectations(self,Fmu,Fvar,Fmuvar,Y):

        return Y * Fmu \
                - tf.exp(Fmu + Fvar / 2 + Fmuvar/2)  \
                - tf.math.lgamma(Y + 1) \
                + Y

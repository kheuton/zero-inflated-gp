from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from gpflow.base import Parameter
from gpflow.utilities import positive
from gpflow.likelihoods import Gaussian
from gpflow.config import default_float

float_type = default_float()
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class OnOffLikelihood(Gaussian):
    """
    Computes\int {log(y|f) [\int p(f,g) dg]} df = \int {log(y|f) [\int p(f|g) p(g) dg]} df
    Where
    p(y) = N(y|A*fmean,sigma2)
    p(f|g) = N(f| diag(gmean)*fmean,diag(gvar)*fvar)
    p(g) = N(gmean,gvar)

    While marginalising gamma, an uncertainity with respect to mean is introduced as a trace term
    (This term is in addition to standard SVGP classification terms)
    """

    def __init__(self, variance: float = 0.01,):
        super().__init__(variance=variance)

    # not implemented logp, conditional_mean and others

    def variational_expectations(self,Fmu,Fvar,Fmuvar,Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(self.variance) \
                - 0.5 * (tf.square(Y - Fmu) + Fvar + Fmuvar) / self.variance

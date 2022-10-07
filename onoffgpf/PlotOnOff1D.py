import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import ticker
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

def normcdf(x):
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1. - 2.e-3) + 1.e-3

def PlotOnOff1D(m, softplus=False):
    mpl.rcParams['figure.figsize'] = (11.0,10.0)
    mpl.rcParams.update({'font.size': 20})

    _X = m.Xtrain
    _Y = m.Ytrain

    _gfmean,_gfvar,_,_fmean,_fvar,_gmean,_gvar,_pgmean,_pgvar = m.predict_onoffgp(_X)
    data_shape = _fmean.shape
    _Zf = m.Zf.numpy()
    _Kf = m.kernf.K(_X)
    _u_fm = m.u_fm.numpy()
    _u_fs_sqrt = m.u_fs_sqrt

    _Zg = m.Zg.numpy()
    _Kg = m.kerng.K(_X)
    _u_gm = m.u_gm.numpy()
    _u_gs_sqrt = m.u_gs_sqrt

    _variance = m.likelihood.variance

    _Kpg = tf.reshape(_pgmean,(-1,1)) * tf.reshape(_pgmean, (1,-1))
    _Kfg = _Kpg * _Kf

    g_mean_NDS = tf.expand_dims(_gmean, -1)
    g_std_NDS = tf.expand_dims(tf.math.sqrt(_gvar), -1)
    f_mean_NDS = tf.expand_dims(_fmean, -1)
    f_var_NDS = tf.expand_dims(_fvar, -1)
    f_std_NDS = tf.math.sqrt(f_var_NDS)
    _X = _X.flatten()
    _Y = _Y.flatten()
    _gfmean = tf.reshape(_gfmean, -1)
    _gfvar  = tf.reshape(_gfvar, -1)
    _fmean  = tf.reshape(_fmean, -1)
    _fvar   = tf.reshape(_fvar, -1)
    _gmean  = tf.reshape(_gmean, -1)
    _gvar   = tf.reshape(_gvar, -1)
    _pgmean = tf.reshape(_pgmean, -1)
    _pgvar  = tf.reshape(_pgvar, -1)


    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[0, 0:-1])
    ax2 = plt.subplot(gs[1, 0:-1])
    ax3 = plt.subplot(gs[2, 0:-1])
    ax4 = plt.subplot(gs[3, 0:-1])
    ax5 = plt.subplot(gs[0, -1])
    ax6 = plt.subplot(gs[1, -1])
    ax7 = plt.subplot(gs[2, -1])
    ax8 = plt.subplot(gs[3, -1])


    # plot y

    if softplus:

        u = tf.random.normal(shape=data_shape + (m.samples,))
        w = tf.random.normal(shape=data_shape + (m.samples,))

        # Expand dims to give the mean a sample dimension
        g_samples = g_mean_NDS + u * g_std_NDS
        del u
        phi_g_samples = normcdf(g_samples)
        del g_samples

        f_samples = f_mean_NDS * phi_g_samples + w * f_std_NDS * phi_g_samples

        shifted_softplus_f_samples = phi_g_samples*tf.math.softplus(f_samples + 2)

        y_poi = tfp.distributions.Poisson(rate=shifted_softplus_f_samples,
                                          force_probs_to_zero_outside_support=True)

        y_samples = y_poi.sample()

        shifted_softplus_f = tf.squeeze(tf.reduce_mean(shifted_softplus_f_samples, -1))
        shifted_softplus_f_up = tf.squeeze(tfp.stats.percentile(y_samples, 97.5, axis=-1))
        shifted_softplus_f_low = tf.squeeze(tfp.stats.percentile(y_samples, 2.5, axis=-1))
        ax1.plot(_X, shifted_softplus_f, '-', color='#ff7707')
        y1 = shifted_softplus_f_up
        y2 = shifted_softplus_f_low
    else:
        ax1.plot(_X, _gfmean, '-', color='#ff7707')
        y1 = (_gfmean-1.5*((np.sqrt(_fvar) * _pgmean + np.sqrt(_pgvar)*(1-_pgmean)) + np.sqrt(_variance)))
        y2 = (_gfmean+1.5*((np.sqrt(_fvar) * _pgmean +  np.sqrt(_pgvar)*(1-_pgmean)) + np.sqrt(_variance)))
    ax1.fill_between(_X,y1,y2,facecolor='#ff7707',alpha=0.5)
    ax1.scatter(_X,_Y,s=8,
             color='black',alpha=0.7)
    ax1.set_xlim(0,10)
    # ax1.set_ylabel("Data" + r"$\mathbf{y}|\mathbf{f}$")
    ax1.set_title("(a) Predictive function \n"+r"$\mathbf{y}$",fontsize=18)
    ax1.set_yticks([-1,0,1], [])
    ax1.set_xticks([], [])

    # plot f and f|g
    ax2.plot(_X,_fmean,'-',color='#008b62',label=r"$f$")
    if softplus:
        f1 = tf.squeeze(tfp.stats.percentile(shifted_softplus_f_samples, 97.5, axis=-1))
        f2 = tf.squeeze(tfp.stats.percentile(shifted_softplus_f_samples, 2.5, axis=-1))

    else:
        f1 = (_fmean-1.5*np.sqrt(_fvar))
        f2 = (_fmean+1.5*np.sqrt(_fvar))
        ax2.fill_between(_X,f1,f2,facecolor='#008b62',alpha=0.5)
    ax2.plot(_Zf,_u_fm,
             marker='o',linestyle = 'None',
             markeredgecolor = 'None',
             markerfacecolor='#008b62',alpha=0.7) #,label = 'uf (optimized)')

    if softplus:
        ax2.plot(_X, shifted_softplus_f, '-', color='#ff7707', label=r"$f|g$")
        f3 = shifted_softplus_f_up
        f4 = shifted_softplus_f_low
    else:
        ax2.plot(_X,_gfmean,'-',color='#ff7707',label=r"$f|g$")
        f3 = (_gfmean-1.5*(np.sqrt(_fvar) * _pgmean + np.sqrt(_pgvar)*(1-_pgmean)))
        f4 = (_gfmean+1.5*(np.sqrt(_fvar) * _pgmean + np.sqrt(_pgvar)*(1-_pgmean)))
    ax2.fill_between(_X,f3,f4,facecolor='#ff7707',alpha=0.5)

    ax2.set_xlim(0,10)
    # ax2.set_ylabel("Augmnted "+ r"$\mathbf{f}|\mathbf{g}$",fontsize=18)
    ax2.set_xticks([], [])
    ax2.set_yticks([-1,0,1], [])
    ax2.set_title("(c) Sparse latent function \n"+ r"$\mathbf{f}|\mathbf{g}$",fontsize=18)
    ax2.legend(loc="lower right",ncol=1,fontsize=18)

    # plot phi(gamma)
    ax3.plot(_X,_pgmean,'-',color='#003366')
    if softplus:
        pg1 = tf.squeeze(tfp.stats.percentile(phi_g_samples, 97.5, axis=-1))
        pg2 = tf.squeeze(tfp.stats.percentile(phi_g_samples, 2.5, axis=-1))
    else:
        pg1 = (_pgmean-2*np.sqrt(_pgvar))
        pg2 = (_pgmean+2*np.sqrt(_pgvar))
    ax3.fill_between(_X,pg1,pg2,facecolor='#6684a3',alpha=0.7)
    ax3.axhline(y=0.5,linestyle='--',color='#333333')
    ax3.set_xlim(0,10)
    ax3.set_title("(e) Probit support function \n" + r"$\Phi(\mathbf{g})$",fontsize=18)
    ax3.set_xticks([], [])
    # ax3.set_ylabel("Support " + r"$\Phi(\mathbf{g})$",fontsize=18)

    # plot gamma
    ax4.plot(_X,_gmean,'-',color='#003366')
    ax4.plot(_Zg,_u_gm,
             marker='o',linestyle = 'None',
             markeredgecolor = 'None',
             markerfacecolor='#003366',alpha=0.8) #,label = 'ug (optimized)')
    g1 = (_gmean-2*np.sqrt(_gvar))
    g2 = (_gmean+2*np.sqrt(_gvar))
    ax4.fill_between(_X,g1,g2,facecolor='#6684a3',alpha=0.7)
    # ax4.set_ylabel("Latent  " +r"$\mathbf{g}$",fontsize=18)
    ax4.axhline(y=0.0,linestyle='--',color='#333333')
    ax4.set_title("(g) Latent function \n" +r"$\mathbf{g}$",fontsize=18)
    ax4.set_xlim(0,10)
    # plt.title('kernel lengthscale = %.3f, variance = %.3f' % (klg,ksg))

    im5 = ax5.imshow(_Kfg,cmap="viridis")
    cb = plt.colorbar(im5,ax=ax5,fraction=0.046, pad=0.03,extend="max")
    ax5.set_title("(b) Sparse kernel \n" + r"$\Phi(\mathbf{g}) \Phi(\mathbf{g})^T \circ K_f$",fontsize=18)
    ax5.set_xticks([], [])
    ax5.set_yticks([], [])
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()


    im6 = ax6.imshow(_Kf,cmap="viridis")
    cb = plt.colorbar(im6,ax=ax6,fraction=0.046, pad=0.03,extend="max")
    ax6.set_title("(d) Latent kernel \n"+r"$K_f$",fontsize=18)
    ax6.set_xticks([], [])
    ax6.set_yticks([], [])
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()


    im7 = ax7.imshow(_Kpg,cmap="viridis")
    cb = plt.colorbar(im7,ax=ax7,fraction=0.046, pad=0.03,extend="max")
    ax7.set_title("(f) Probit kernel \n" + r"$\Phi(\mathbf{g}) \Phi(\mathbf{g})^T$",fontsize=18)
    ax7.set_xticks([], [])
    ax7.set_yticks([], [])
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()


    im8 = ax8.imshow(_Kg,cmap="viridis")
    cb =  plt.colorbar(im8,ax=ax8,fraction=0.046, pad=0.03,extend="max")
    ax8.set_title("(h) Latent kernel \n"+r"$K_g$",fontsize=18)
    ax8.set_xticks([], [])
    ax8.set_yticks([], [])
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.5,wspace=0.1)
    plt.savefig("plots/toy.png")
    plt.show()

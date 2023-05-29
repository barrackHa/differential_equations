import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim, rc
from matplotlib.collections import LineCollection
from scipy.stats import linregress
from ews_helper import get_ews, itoEulerMaruyama
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

import warnings
warnings.filterwarnings("error")
              
def annotate_axes(ax, text, fontsize=12):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=fontsize, color="darkgrey")

def set_axes_title(ax, text, fontsize=12):
    ax.title.set_text(text)
    ax.title.fontsize = fontsize

def hopf_model(t, S, epsilon, omega, b):
    """S = [dr/dt, dtheta/dt, dmu/dt]"""
    r, theta, mu = S
    try:
        shaper = np.ones_like(mu)
    except:
        try:
            shaper = np.ones(len(mu))
        except:
            shaper = 1

    return [
        (mu*r) - (r**3),
        omega + (b*(r**2)),
        epsilon * shaper
    ]

def calc_and_plot_hopf_maruyama(
        ax, time, r_0, theta_0, mu_0, epsilon, noise, b, 
        save_derivative=True, plot_strech=1.2, 
        polar_to_cartisian=True, color='b', label='Hopf'):
    """
    Plot the hopf
    Returns t_star, ax, results, derivatives
    """

    results, derivatives = itoEulerMaruyama(
        hopf_model,
        y0=[r_0, theta_0, mu_0],
        time=time,noise=[0, noise, 0],
        args=(epsilon, omega, b), save_derivative=save_derivative
    )

    rs, thetas, mus = results[:,0], results[:,1], results[:,2]
    if polar_to_cartisian:
        x = rs*np.cos(thetas)
        y = rs*np.sin(thetas)
    else:
        x = rs
        y = thetas

    slope,intercept,_,_,_ = linregress(time, mus)
    # mu=0=intercept+slope*t iff t=-intercept/slope
    t_star = -intercept/slope

    w = plot_strech
    ax.plot(x,y, label=label, c=color)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x.min()*w, x.max()*w)
    ax.set_ylim(y.min()*w, y.max()*w)
    ax.scatter(x[0], y[0], c='g', s=50, label='start')
    ax.scatter(x[-1], y[-1], c='r', s=50, label='end')
    ax.legend()

    return t_star, ax, results, derivatives

def calc_and_plot_ews(
        axs, time, arr, ews_win_size=24, ews_offset=1, 
        label=None, t_star=None, 
        block_idxs=None, ar1s=None, decays=None, vars=None):
    ax0, ax1, ax2 = axs
    ax0.set_ylabel('AR1')
    ax1.set_ylabel('Acorr Decay Times')
    ax1.sharex(ax0)
    ax2.set_ylabel('Variance')
    ax2.sharex(ax0)
    ax2.set_xlabel('Time')
    
    if ar1s is None or decays is None or vars is None or block_idxs is None:
        block_idxs, ar1s, decays, vars = get_ews(
                time, arr, 
                win_size=ews_win_size, offset=ews_offset
        )
    ax0.plot(
        time[block_idxs[:len(ar1s)]], ar1s, label=label
    )
    ax1.plot(
        time[block_idxs[:len(decays)]], decays, label=label
    )
    ax2.plot(
        time[block_idxs[:len(vars)]], vars, label=label
    )

    for ax in axs:
        ax.legend()
        if t_star is not None:
            ax.axvline(
                t_star, color='k', linestyle='--', 
                alpha=0.5, label='mu=0'
            )
    return fig, [ax0, ax1, ax2], block_idxs, ar1s, decays, vars

def plot_pca_analysis(x, y, title):
    pca = PCA(n_components=(2))
    pca_input = np.array([x, y]).T
    pca_input = StandardScaler().fit_transform(pca_input)
    principalComponents = pca.fit_transform(pca_input).T
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title, fontsize=10)
    spec = fig.add_gridspec(4, 2)
    ax00, ax01 = fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1])
    axs = [fig.add_subplot(spec[i, :]) for i in range(1,4)]
    ax00.plot(principalComponents[0], principalComponents[1], label='PC\'s', c='b')
    set_axes_title(ax00, 'PC Plane', fontsize=4)
    ax00.set_xlabel('PC1')
    ax00.set_ylabel('PC2')
    ax01.barh(['PC1','PC2'], pca.explained_variance_ratio_, color='orange')
    set_axes_title(ax01, 'PCA Explained Variance Ratio', fontsize=4)
    ax01.invert_yaxis()

    fig, axs, block_idxs, ar1s, decays, vars = calc_and_plot_ews(
        axs, time, principalComponents[0], 
        ews_win_size, ews_offset, label='PC1', t_star=t_star
    )

    fig, axs, block_idxs, ar1s, decays, vars = calc_and_plot_ews(
        axs, time, principalComponents[1], 
        ews_win_size, ews_offset, label='PC2', t_star=t_star
    )
    return fig, axs


if __name__ == '__main__':

    r_0 = 0.1
    omega = 2
    b = 1
    theta_0 = 0
    mu_0 = -2.0
    epsilon = 0.01
    sigma = 0.9
    t_span, t_points= 410, 50000 
    time = np.linspace(0, t_span, t_points)
    dt = t_span/t_points
    ews_win_size, ews_offset = 21, 1

    # Plot Hopf sim solutions
    fig = plt.figure(figsize=(16, 8))
    ttl = f"""Hopf Model: df/dt = [(mu*r)-(r**3), {omega}+({b}*(r**2))+noise, {epsilon}]
    [r_0, theta_0, mu_0]=[{r_0}, {theta_0}, {mu_0}], dt={dt}[Sec]"""
    fig.suptitle(ttl, fontsize=10)
    spec = fig.add_gridspec(3, 2)

    # Hopf with noise
    ax00 = fig.add_subplot(spec[0, 0])
    noisy_hopf = calc_and_plot_hopf_maruyama(
        ax00, time, r_0, theta_0, mu_0, epsilon, noise=sigma, b=b,
        label=None, color='b'
    )
    t_star = noisy_hopf[0]
    set_axes_title(ax00, f'Hopf With noise (={sigma})', fontsize=4)

    # Hopf without noise
    ax01 = fig.add_subplot(spec[0, 1])
    no_noise = calc_and_plot_hopf_maruyama(
        ax01, time, r_0, theta_0, mu_0, epsilon, noise=0, b=b,
        label=None, color='orange'
    )
    set_axes_title(ax01, 'Hopf Without noise', fontsize=4)

    ax1 = fig.add_subplot(spec[1, :])
    set_axes_title(ax1, 'Solutions as a function of time', fontsize=6)
    ax1.set_facecolor(plt.cm.gray(.95))
    ax1.set_ylabel('F(t)')

    for i in [0,2]:
        ax1.plot(time, no_noise[2][:,i], label='r' if i==0 else 'mu')
    ax1.axvline(no_noise[0], color='k', linestyle='--', alpha=0.5, label='mu=0')
    
    ax2 = fig.add_subplot(spec[2, :], sharex=ax1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Theta(t)')
    ax2.set_facecolor(plt.cm.gray(.95))
    ax2.plot(time, no_noise[2][:,1], label=f'Theta', c='g')

    for ax in [ax1, ax2]:
        ax.legend()
        ax.grid()

    fig.savefig(Path(__file__).parents[0]/f'tmp_figs/hopf.png')

    ## EWS Thetas 
    
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('EWS Of Thetas W/O Noise (Hopf)', fontsize=10)
    spec = fig.add_gridspec(3, 2)
    axs = [fig.add_subplot(spec[i, :]) for i in range(3)]
    # With noise
    fig, axs, block_idxs, ar1s, decays, vars = calc_and_plot_ews(
        axs, time, noisy_hopf[2][:,1], 
        ews_win_size, ews_offset, label='Noised Thetas'
    )
    # No noise
    fig, axs, block_idxs, ar1s, decays, vars = calc_and_plot_ews(
        axs, time, no_noise[2][:,1], 
        ews_win_size, ews_offset, label='No noise'
    )
    fig.savefig(Path(__file__).parents[0]/f'tmp_figs/thetas.png')    
    
    x = no_noise[2][:,0]*np.cos(no_noise[2][:,1])
    y = no_noise[2][:,0]*np.sin(no_noise[2][:,1])
    noisy_x = noisy_hopf[2][:,0]*np.cos(noisy_hopf[2][:,1])
    noisy_y = noisy_hopf[2][:,0]*np.sin(noisy_hopf[2][:,1])
    
    ## EWS For X-Y W/O Noise
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('EWS Of X\'s W/O Noise (Hopf)', fontsize=10)
    spec = fig.add_gridspec(3, 2)
    axs = [fig.add_subplot(spec[i, :]) for i in range(3)]
    # X W/O noise
    fig, axs, block_idxs, ar1s, decays, vars = calc_and_plot_ews(
        axs, time, noisy_x, 
        ews_win_size, ews_offset, label='noisy_x', t_star=t_star
    )
    fig, axs, block_idxs, ar1s, decays, vars = calc_and_plot_ews(
        axs, time, x, 
        ews_win_size, ews_offset, label='x'
    )
    fig.savefig(Path(__file__).parents[0]/f'tmp_figs/xs.png')
    
    # Y W/O noise
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('EWS Of Y\'s W/O Noise (Hopf)', fontsize=10)
    spec = fig.add_gridspec(3, 2)
    axs = [fig.add_subplot(spec[i, :]) for i in range(3)]
    fig, axs, block_idxs, ar1s, decays, vars = calc_and_plot_ews(
        axs, time, noisy_y, 
        ews_win_size, ews_offset, label='noisy_x', t_star=t_star
    )
    fig, axs, block_idxs, ar1s, decays, vars = calc_and_plot_ews(
        axs, time, y, 
        ews_win_size, ews_offset, label='y'
    )
    fig.savefig(Path(__file__).parents[0]/f'tmp_figs/ys.png')
    

    # PCA X-Y 
    fig, axs = plot_pca_analysis(x, y, 'EWS Of X&Y Without Noise (Hopf)')
    fig.savefig(Path(__file__).parents[0]/f'tmp_figs/pca_no_noise.png')
    # PCA X-Y with noise
    fig, axs = plot_pca_analysis(
        noisy_x, noisy_y, 'EWS Of X&Y Without Noise (Hopf)'
    )
    fig.savefig(Path(__file__).parents[0]/f'tmp_figs/pca_noisy.png')
    
    plt.show()

# with noise / without noise
# AR1, decay time, var
# [x, y], [r, theta, mu], PCA
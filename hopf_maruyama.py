import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim, rc
from matplotlib.collections import LineCollection
from scipy.stats import linregress
from ews_helper import get_ews, itoEulerMaruyama

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



if __name__ == '__main__':

    r_0 = 0.1
    omega = 2
    b = 1
    theta_0 = 0
    mu_0 = -2.0
    epsilon = 0.01
    sigma = 0.9
    time = np.linspace(0, 510, 50000)
    ews_win_size, ews_offset = 21, 1

    # fig, axs = plt.subplots(1,2)
    fig = plt.figure(figsize=(16, 8))
    spec = fig.add_gridspec(3, 2)

    # Hopf with noise
    ax00 = fig.add_subplot(spec[0, 0])
    noisy_hopf = calc_and_plot_hopf_maruyama(
        ax00, time, r_0, theta_0, mu_0, epsilon, noise=sigma, b=b,
        label=None, color='b'
    )
    set_axes_title(ax00, 'Hopf With noise', fontsize=8)

    # Hopf without noise
    ax01 = fig.add_subplot(spec[0, 1])
    no_noise = calc_and_plot_hopf_maruyama(
        ax01, time, r_0, theta_0, mu_0, epsilon, noise=0, b=b,
        label=None, color='orange'
    )
    set_axes_title(ax01, 'Hopf Without noise', fontsize=8)

    ax1 = fig.add_subplot(spec[1, :])
    ax1.set_facecolor(plt.cm.gray(.95))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('AR1')

    """
    # EWS of Hopf with noise
    block_idxs, noisy_ar1s = map(
        get_ews(
            time, noisy_hopf[2][:,1], 
            win_size=ews_win_size, offset=ews_offset
        ).get,
        ['block_idxs', 'ar1s']
    )
    ax1.plot(
        time[block_idxs[:len(noisy_ar1s)]], noisy_ar1s, label=f'Noised Thetas'
    )

    # EWS of Hopf without noise
    ar1s = get_ews(
        time, no_noise[2][:,1], win_size=ews_win_size, offset=ews_offset
    )['ar1s']    
    ax1.plot(time[block_idxs[:len(ar1s)]], ar1s, label=f'No Noise Thetas')
    
    # ax1.axvline(no_noise[0], color='k', linestyle='--', alpha=0.5, label='t*')
    ax1.axvline(noisy_hopf[0], color='k', linestyle='--', alpha=0.5, label='noisy_t*')
    ax1.legend()
    set_axes_title(ax1, 'AR1 of Thetas', fontsize=8)
    """

    ax2 = fig.add_subplot(spec[2, :], sharex=ax1)
    ax2.set_facecolor(plt.cm.gray(.95))
    # ax2.set_xlabel('Time')
    ax1.set_ylabel('AR1')
    # for i in [0,2]:
    #     ax2.plot(time, no_noise[2][:,i], label=f'results[:, {i}]')
    # ax2.axvline(no_noise[0], color='k', linestyle='--', alpha=0.5, label='t*')
    # ax2.legend()

    x = no_noise[2][:,0]*np.cos(no_noise[2][:,1])
    y = no_noise[2][:,0]*np.sin(no_noise[2][:,1])
    noisy_x = noisy_hopf[2][:,0]*np.cos(noisy_hopf[2][:,1])
    noisy_y = noisy_hopf[2][:,0]*np.sin(noisy_hopf[2][:,1])


    block_idxs, ar1s = map(
        get_ews(
            time, x, 
            win_size=ews_win_size, offset=ews_offset
        ).get,
        ['block_idxs', 'ar1s']
    )  
    ax1.plot(time[block_idxs[:len(ar1s)]], ar1s, label=f'x')

    ar1s = get_ews(
        time, y, win_size=ews_win_size, offset=ews_offset
    )['ar1s']    
    ax1.plot(time[block_idxs[:len(ar1s)]], ar1s, label=f'y')

    block_idxs, ar1s = map(
        get_ews(
            time, noisy_x, 
            win_size=ews_win_size, offset=ews_offset
        ).get,
        ['block_idxs', 'ar1s']
    )  
    ax2.plot(time[block_idxs[:len(ar1s)]], ar1s, label=f'noisy_x')

    ar1s = get_ews(
        time, noisy_y, win_size=ews_win_size, offset=ews_offset
    )['ar1s']    
    ax2.plot(time[block_idxs[:len(ar1s)]], ar1s, label=f'noisy_y')
    ax2.legend()
        
    plt.show()

# with noise / without noise
# AR1, decay time, var
# [x, y], [r, theta, mu], PCA
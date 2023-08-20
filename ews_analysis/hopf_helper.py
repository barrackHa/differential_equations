import numpy as np
from scipy.stats import linregress
from ews_analysis.ews_helper import get_ews, itoEulerMaruyama

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
        ax, time, r_0, omega, theta_0, mu_0, epsilon, noise, b, 
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
        fig, axs, time, arr, ews_win_size=24, ews_offset=1, 
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
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # ax.legend(loc='outside upper right')
        if t_star is not None:
            ax.axvline(
                t_star, color='k', linestyle='--', 
                alpha=0.5, label='mu=0'
            )
    return fig, [ax0, ax1, ax2], block_idxs, ar1s, decays, vars


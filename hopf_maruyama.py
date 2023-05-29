import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim, rc
from matplotlib.collections import LineCollection
from scipy.stats import linregress

def annotate_axes(ax, text, fontsize=12):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=fontsize, color="darkgrey")


def itoEulerMaruyama(model, y0, time, noise, args=None, save_derivative=False):
    ret_val = np.zeros((len(time), len(y0)))
    noise = np.array(noise)
    y0 = np.array(y0)
    ret_val[0, :] = y0
    dt = time[1] - time[0]
    derivatives = np.zeros((len(time), len(y0)))
    for i in range(1, len(time)):
        derivatives[i-1, :] = np.array(
            model(ret_val[i - 1, :], time[i], *args) 
                if args else model(ret_val[i - 1, :], time[i])
        ) 
        ret_val[i, :] = ret_val[i - 1, :] + \
                        derivatives[i-1, :] * dt + \
                        noise*np.random.normal(0,np.sqrt(dt),ret_val.shape[1])
        
    return ret_val if not save_derivative else (ret_val,derivatives)

def hopf_model(S, t, epsilon, omega, b):
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


if __name__ == '__main__':

    r_0 = 0.1
    omega = 2
    b = 1
    theta_0 = 0
    mu_0 = -2.0
    epsilon = 0.01
    sigma = 0.9
    time = np.linspace(0, 410, 50000)

    # fig, axs = plt.subplots(1,2)
    fig = plt.figure(figsize=(16, 8))
    spec = fig.add_gridspec(2, 2)
    w = 1.2

    nosiy_results, nosiy_derivatives = itoEulerMaruyama(
        hopf_model,
        y0=[r_0, theta_0, mu_0],
        time=time,noise=[0, sigma, 0],
        args=(epsilon, omega, b),save_derivative=True
    )
    nosiy_rs, nosiy_thetas = nosiy_results[:,0], nosiy_results[:,1]
    nosiy_x = nosiy_rs*np.cos(nosiy_thetas)
    nosiy_y = nosiy_rs*np.sin(nosiy_thetas)
    
    ax00 = fig.add_subplot(spec[0, 0])
    annotate_axes(ax00, 'with noise')
    ax00.plot(nosiy_x,nosiy_y, label='with noise')
    ax00.set_xlim(nosiy_x.min()*w, nosiy_x.max()*w)
    ax00.set_ylim(nosiy_y.min()*w, nosiy_y.max()*w)
    ax00.scatter(nosiy_x[0], nosiy_y[0], c='g', s=50, label='start')
    ax00.scatter(nosiy_x[-1], nosiy_y[-1], c='r', s=50, label='end')
    ax00.legend()

    # Without noise
    results,derivatives = itoEulerMaruyama(
        hopf_model,
        y0=[r_0, theta_0, mu_0],
        time=time,noise=[0,0,0],
        args=(epsilon, omega, b),save_derivative=True
    )
    rs, thetas=results[:,0],results[:,1]
    x = rs*np.cos(thetas)
    y = rs*np.sin(thetas)
    
    ax01 = fig.add_subplot(spec[0, 1])
    annotate_axes(ax01, 'without noise')
    ax01.plot(x,y, label='without noise', c='orange')
    ax01.set_xlim(x.min()*w,x.max()*w)
    ax01.set_ylim(y.min()*w,y.max()*w)
    ax01.scatter(x[0], y[0], c='g', s=50, label='start')
    ax01.scatter(x[-1], y[-1], c='r', s=50, label='end')
    ax01.legend()

    # ax1 = fig.add_subplot(spec[1, :])
    # annotate_axes(ax1, 'SOl as a function of time')
    # for i, name in enumerate(['r','theta','mu']):
    #     ax1.plot(time,results[:,i], label=name)
    # ax1.plot(time,x, label='x(t)')
    # ax1.plot(time,y, label='y(t)', linestyle='-.')
    # ax1.set_xlim(time.min(),time.max())
    # ax1.grid()
    # ax1.legend()
    # print(np.allclose(x,y, rtol=1e-1, atol=1))
    # print(x[:5], y[:5])

    ax1 = fig.add_subplot(spec[1, :])
    ax1.set_facecolor(plt.cm.gray(.95))
    win_size = 21 
    for i in range (1, 2):
        ar1s = []
        noisy_ar1s = []
        offset = i
        block_idxs = np.arange(time.shape[0]-win_size, step=offset)
        
        for i in block_idxs:
            try:
                lag0 = thetas[i: i+win_size]
                lag1 = thetas[i+offset: i+offset+win_size]
                n_lag0 = nosiy_thetas[i: i+win_size]
                n_lag1 = nosiy_thetas[i+offset: i+offset+win_size]
                ar1s.append(np.corrcoef(lag0, lag1)[0, 1])
                noisy_ar1s.append(np.corrcoef(n_lag0, n_lag1)[0, 1])
            except Exception as e:
                print(f'error in {i}\n{e}')
    
    print(block_idxs[:len(ar1s)].shape, len(ar1s))
    ax1.plot(time[block_idxs[:len(ar1s)]], ar1s, label=f'Normal Form')
    ax1.plot(time[block_idxs[:len(ar1s)]], noisy_ar1s, label=f'With Noise')
    ax1.legend()
        
    plt.show()

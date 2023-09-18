import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from cdp_ews.helpers import itoEulerMaruyama
from helpers import itoEulerMaruyama

def plot_bif_sim(fig, axs, time, results, derivatives):

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    
    axs[0].plot(time, results[:,0]) # , label='x')
    # axs[0].plot(time, derivatives[:,0]) # , label='x')
    # axs[0].plot(time, results[:,1]) #, label='r')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('x')
    # make ax grey
    axs[0].set_facecolor(plt.cm.gray(.85))

    # plot t_star, where r(t) = 0 = r0 + epsilon*t 
    t_star = time[np.where(results[:,1] >= 0)[0][0]]
    # axs[0].axvline(t_star, c='r', ls='--') #label=f't*={t_star:.2f}'

    # ax.set_ylim(-20, 70)
    # axs[0].legend()
    axs[0].grid()
    # plt.show()
    return fig, axs, t_star


def simulate_ts(ode_model, time, y0, noise, args):
    
    results,derivatives = itoEulerMaruyama(
        model=ode_model,
        y0=y0,
        time=time,
        noise=noise, args=args, save_derivative=True
    )
    return results,derivatives

# Pitchfork bifurcation model

def super_pitchfork(t, S, epsilon, a=1):
    """
        x_dot = rx - ax^3
        r_dot = epsilon
    """
    x, r = S
    try:
        shaper = np.ones_like(r)
    except:
        try:
            shaper = np.ones(len(r))
        except:
            shaper = 1

    return [
        r*x - a*x**3,
        epsilon * shaper
    ]

def saddlenode_yuval(t, S, epsilon, a):
    x, y = S
    try:
        shaper = np.ones_like(y)
    except:
        try:
            shaper = np.ones(len(y))
        except:
            shaper = 1

    return [
        y + a*x**2 - x**3,
        epsilon * shaper
    ]

def simulate_super_pitchfork(time, y0, epsilon, a, noise):
        
    results, derivatives = simulate_ts(
        ode_model=super_pitchfork,
        time=time,
        y0=y0,
        noise=noise, args=(epsilon, a)
    )

    return results,derivatives

def simulate_saddlenode(time, y0, epsilon, a, noise):
        
    results, derivatives = simulate_ts(
        ode_model=saddlenode_yuval,
        time=time,
        y0=y0,
        noise=noise, args=(epsilon, a)
    )

    return results,derivatives

def sim_n_plot_sd(fig, axs, time, r0, x0, epsilon, sigma_noise, a=1, ttl=''):

    # Run sim and get data
    results, derivatives = simulate_saddlenode(
        time, y0=[x0, r0], epsilon=epsilon, a=a, noise=[sigma_noise,0]
    )
    
    # Plot
    fig, axs, t_star = plot_bif_sim(fig, axs, time, results, derivatives)

    # Set title    
    tmp_dt = np.mean(time[1:] - time[:-1])
    ttl = ttl + f"time=[{time[0]:.2f},{time[-1]:.2f}], dt={tmp_dt:.2f}, r0={r0},  "
    ttl = ttl + f"x0={x0}, epsilon={epsilon}, sigma_noise={sigma_noise}, a={a}"
    fig.suptitle(ttl, fontsize=10)

    return fig, axs, t_star, results, derivatives

def sim_n_plot_pitchfork(fig, axs, time, r0, x0, epsilon, sigma_noise, a=1, ttl=''):

    # Run sim and get data
    results, derivatives = simulate_super_pitchfork(
        time, y0=[x0, r0], epsilon=epsilon, a=a, noise=[sigma_noise,0]
    )
    
    # Plot
    fig, axs, t_star = plot_bif_sim(fig, axs, time, results, derivatives)

    # Set title    
    tmp_dt = np.mean(time[1:] - time[:-1])
    ttl = ttl + f"time=[{time[0]:.2f},{time[-1]:.2f}], dt={tmp_dt:.2f}, r0={r0},  "
    ttl = ttl + f"x0={x0}, epsilon={epsilon}, sigma_noise={sigma_noise}, a={a}"
    fig.suptitle(ttl, fontsize=10)
    
    return fig, axs, t_star, results, derivatives


class Model():
    
    SUPER_CRITICAL_PITCHFORK = 'SUPER_CRITICAL_PITCHFORK'
    SADDLENODE_EWS_PAPER_MODEL = 'SADDLENODE_EWS_PAPER_MODEL'
    
    def __init__(self, *arg, **kwargs) -> None:
        
        self.time = None
        self.results = None
        self.derivatives = None
        self.t_star = None
        self.epsilon = None
        self.a = None
        self.sigma_noise = None
        self.y0 = None
        self.ttl = None
        self.model = None
        self.model_simulator = None
        self.config = None

    def simulate(self, **kwargs):
        config = kwargs['config']
        self.config = config
        try:
            self.time = config['time']
        except KeyError as e:
            start, stop , dt = config['start'], config['stop'], config['dt']
            self.time = np.arange(start, stop, dt)
        
        self.epsilon = config['epsilon']
        self.sigma_noise = config['sigma_noise']
        self.a = config['a']
        self.y0 = config['y0']

        results, derivatives = simulate_ts(
            # ode_model=saddlenode_yuval,
            ode_model=self.model_simulator,
            time=self.time,
            y0=self.y0,
            noise=self.sigma_noise, args=(self.epsilon, self.a)
        )
        self.results = results
        self.derivatives = derivatives
        
        return results, derivatives
        
    
    def plot(self, fig, axs, **kwargs):
        fig, axs, t_star = plot_bif_sim(
            fig, axs, self.time, self.results, self.derivatives
        )
        # Set title
        t = self.time
        tmp_dt = np.mean(t[1:] - t[:-1])
        x0, r0 = self.y0
        ttl = self.ttl + f"time=[{t[0]:.2f},{t[-1]:.2f}], dt={tmp_dt:.2f}, r0={r0},  "
        ttl = ttl + f"x0={x0}, epsilon={self.epsilon}, sigma_noise={self.sigma_noise}, a={self.a}"
        fig.suptitle(ttl, fontsize=10)

        self.t_star = t_star
        return fig, axs, t_star
    
    def simulate_and_plot(self, **kwargs):
        self.simulate(**kwargs)
        self.plot(**kwargs)

class SuperCriticalPitchfork(Model):

    def __init__(self, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.model = Model.SUPER_CRITICAL_PITCHFORK
        self.model_simulator = super_pitchfork

    def simulate(self, **kwargs):
        config=kwargs['config']
        super().simulate(config=config)
        self.ttl = config['ttl']
        return self.results, self.derivatives
    
    def plot(self, fig, axs, **kwargs):
        fig, axs, t_star = super().plot(fig, axs, **kwargs)

        return fig, axs, t_star
    
    def simulate_and_plot(self, **kwargs):
        self.simulate(**kwargs)
        self.plot(**kwargs)
        
class SaddleNodeEwsPaperModel(Model):
    def __init__(self, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.model = Model.SADDLENODE_EWS_PAPER_MODEL
        self.model_simulator = saddlenode_yuval

    def simulate(self, **kwargs):
        config=kwargs['config']
        super().simulate(config=config)
        self.ttl = config['ttl']
        return self.results, self.derivatives
    
    def plot(self, fig, axs, **kwargs):
        fig, axs, t_star = super().plot(fig, axs, **kwargs)

        return fig, axs, t_star
    
    def simulate_and_plot(self, **kwargs):
        self.simulate(**kwargs)
        self.plot(**kwargs)
    
    
    
if __name__ == '__main__':
    
    ## Pitchfork
    time = np.arange(0, 9, 0.001)
    r0 = -3
    x0 = 0.1
    a = 25
    epsilon = 1.5
    sigma = 0.5
    config = {
        'time': time,
        # 'start': 0,
        # 'stop': 9,
        # 'dt': 0.001,
        'y0': [x0, r0],
        'epsilon': epsilon,
        'sigma_noise': [sigma, 0],
        'a': a,
        'ttl': r"Pitchfork: $\dot{x}$ = rx - $ax^3$,    $\dot{r}$ = $\epsilon$" + "\n" 
    }

    fig, axs = plt.subplots(1,1)

    for i in range(1):
        scp_model = SuperCriticalPitchfork()
        # print(config)
        res, des = scp_model.simulate(config=config)
        print(res.shape, des.shape)
        fig, axs, t_star = scp_model.plot(fig=fig, axs=axs)
    # plt.show()

    fig.clear()
    plt.close()

    ## Saddle Node
    fig_sd, axs_sd = plt.subplots(2,1)
    axs_sd[0].sharex(axs_sd[1])
    time = np.arange(0, 1000, 0.1)
    r0 = -0.5
    x0 = 0.01
    a = 4 #
    epsilon = 0.001
    sigma = 0.04 #

    ttl = r"Saddle Node: $\dot{x}$ = r + a$x^2$ - $x^3$,    $\dot{r}$ = $\epsilon$" + "\n"

    config = {
        'time': time,
        'y0': [x0, r0],
        'epsilon': epsilon,
        'sigma_noise': [sigma, 0],
        'a': a,
        'ttl': ttl
    }


    fig, axs = plt.subplots(1,1)
    for i in range(10):
        sd_model = SaddleNodeEwsPaperModel()
        # print(config)
        res, des = sd_model.simulate(config=config)
        print(res.shape, des.shape)
        fig, axs, t_star = sd_model.plot(fig=fig, axs=axs)
    plt.show()




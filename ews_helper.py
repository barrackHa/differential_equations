import numpy as np
from scipy.interpolate import interp1d

def z_score_normalizer(arr, axis=0):
    """
    Returns a zscores normalized array by axis 
    """
    std = np.std(arr, axis, keepdims=True)
    # Replace all 0 enteries of std with 1 so there's no division by 0 
    std = (std == 0).astype(int) + std
    mean = np.mean(arr, axis, keepdims=True)

    return (arr - mean) / std

def solve_equality(arr, r):
    return np.argmin(np.sign(arr - r))

def get_acorr_decay_time(arr):
    frame_rate = 1
    tmp_a = z_score_normalizer(arr)
    acorr = np.correlate(tmp_a, tmp_a, mode='full') / tmp_a.size

    idx = solve_equality(
        acorr[acorr.size // 2:], 1 / np.e
    )
    yy = acorr[acorr.size // 2:]
    xx = np.arange(yy.size) / frame_rate
    piecewise_linear_acorr = interp1d(xx, yy, kind='linear')

    delta = 0.001
    grained_xx = np.arange(0, idx, delta) / frame_rate
    approximation = piecewise_linear_acorr(grained_xx)
    decay_time = solve_equality(
        approximation, 1 / np.e
    ) * delta / frame_rate

    return decay_time


def get_ews(time, arr, win_size=21, offset=1):
    """
    @Params: time, arr, win_size, offset
    Returns: dict of block_idxs and ar1s
    """ 
    block_idxs = np.arange(time.shape[0]-win_size, step=offset)
    ar1s = np.zeros_like(block_idxs, dtype=np.float64)
    ar_decay_times = np.zeros_like(block_idxs, dtype=np.float64)
    vars = np.zeros_like(block_idxs, dtype=np.float64)

    for i in block_idxs:
        try:
            lag0 = arr[i: i+win_size]
            lag1 = arr[i+offset: i+offset+win_size]
            ar1s[i] = np.corrcoef(lag0, lag1)[0, 1]
            ar_decay_times[i] = get_acorr_decay_time(lag0)
            vars[i] = np.var(lag0)
        except Exception as e:
            print(f'error in {i}\n{e}')
            

    ar_decay_times = ar_decay_times/np.mean(ar_decay_times)
    vars = vars/np.max(vars)
    return block_idxs, ar1s, ar_decay_times, vars

def itoEulerMaruyama(model, y0, time, noise, args=None, save_derivative=False):
    ret_val = np.zeros((len(time), len(y0)))
    noise = np.array(noise)
    y0 = np.array(y0)
    ret_val[0, :] = y0
    dt = time[1] - time[0]
    derivatives = np.zeros((len(time), len(y0)))
    for i in range(1, len(time)):
        derivatives[i-1, :] = np.array(
            model(time[i], ret_val[i - 1, :], *args) 
                if args else model(ret_val[i - 1, :], time[i])
        ) 
        ret_val[i, :] = ret_val[i - 1, :] + \
                        derivatives[i-1, :] * dt + \
                        noise*np.random.normal(0,np.sqrt(dt),ret_val.shape[1])
        
    return ret_val if not save_derivative else (ret_val,derivatives)


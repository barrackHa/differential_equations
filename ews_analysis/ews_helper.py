# %%
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

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
    try:
        frame_rate = 1
        tmp_a = z_score_normalizer(arr)
        if (tmp_a==0).all():
            tmp_a = np.ones_like(tmp_a)
        if tmp_a.size == 0:
            raise Exception('tmp_a.size == 0')
        acorr = np.correlate(tmp_a, tmp_a, mode='full') / (tmp_a.size - 1)

        idx = solve_equality(
            acorr[acorr.size // 2:], 1 / np.e
        )
        yy = acorr[acorr.size // 2:]
        xx = np.arange(yy.size) / frame_rate
        piecewise_linear_acorr = interp1d(xx, yy, kind='linear')
        # todo: find real delta and frame rate
        delta = 0.001
        grained_xx = np.arange(0, idx, delta) / frame_rate
        approximation = piecewise_linear_acorr(grained_xx)
        decay_time = solve_equality(
            approximation, 1 / np.e
        ) * delta / frame_rate
    except Exception as e:
        print(f'error in get_acorr_decay_time\n{e}')
        exit(-1)
        
    return decay_time


def get_ews(time, arr, win_size=21, offset=1):
    """
    @Params: time, arr, win_size, offset
    Returns: dict of block_idxs and ar1s
    """ 
    # block_idxs = np.arange(time.shape[0]-win_size, step=offset)
    n = arr.shape[0]
    block_idxs = np.arange(n-win_size, step=offset)
    ar1s = np.zeros_like(block_idxs, dtype=np.float64)
    ar_decay_times = np.ones_like(block_idxs, dtype=np.float64)
    vars = np.zeros_like(block_idxs, dtype=np.float64)
    

    for j, i in enumerate(tqdm(block_idxs)):
        failed = False
        try:
            lag0 = arr[i: i+win_size]
            # lag1 = arr[i+offset: i+offset+win_size]
        except Exception as e:
            print(f'Failed to splice arraies. Error in #{j}\n{e}')
            failed = True
        try:
            # ar1s[j] = np.corrcoef(lag0[:-1], lag0[1:])[0, 1]
            a = arr[i: i+win_size]
            l = win_size
            tmp_a = a - np.mean(a)
            ar1 = np.correlate(tmp_a, tmp_a, mode='full') / (np.cov(a)* (l-1))
            ar1s[j] = ar1[win_size]
        except Exception as e:
            print(f'Failed to calculate ar1s. Error in #{j}\n{e}')
            # print(f'lags: {lag0.shape}, {lag1.shape}')
            print(f'index: {i}, win_size: {win_size}, offset: {offset}')
            raise e
            # ToDo: use logger instead of print
            # print('Calculating ar1s using covarience instead')
            # try:
            #     c = np.cov(lag0, lag1)
            #     if c[0,1]==0 or c[1,0]==0:
            #         ar1s[i] = c[0,0]
            #     else:
            #         ar1s[i] = c[0,1]/np.sqrt(c[0,0]*c[1,1])
            # except Exception as e:
            #     failed = True   
        try:
            ar_decay_times[j] = get_acorr_decay_time(lag0)
        except Exception as e:
            print(f'Failed to calculate ar_decay_times. Error in #{j}\n{e}')
            failed = True
        try:
            vars[j] = np.var(lag0) 
        except Exception as e:
            print(f'Failed to calculate vars. Error in #{j}\n{e}')
            failed = True
        if failed: 
            print('Failed during EWS calculate')
            

    ar_decay_times = ar_decay_times/np.mean(ar_decay_times)
    vars = vars/np.max(vars)
    return block_idxs, ar1s, ar_decay_times, vars

def itoEulerMaruyama(model, y0, time, noise, args=None, save_derivative=False):
    ret_val = np.zeros((len(time), len(y0)))
    noise = np.array(noise)
    y0 = np.array(y0)
    ret_val[0, :] = y0
    # print(y0)
    # dt = time[1] - time[0]
    derivatives = np.zeros((len(time), len(y0)))
    for i in tqdm(range(1, len(time))):
        dt = time[i] - time[i-1]
        derivatives[i-1, :] = np.array(
            model(time[i], ret_val[i - 1, :], *args) 
                if args else model(ret_val[i - 1, :], time[i])
        ) 
        ret_val[i, :] = ret_val[i - 1, :] + \
                        derivatives[i-1, :] * dt + \
                        noise*np.random.normal(0,np.sqrt(dt),ret_val.shape[1])
        
    return ret_val if not save_derivative else (ret_val,derivatives)


# %%

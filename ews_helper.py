import numpy as np

def get_ews(time, arr, win_size=21, offset=1):
    """
    @Params: time, arr, win_size, offset
    Returns: dict of block_idxs and ar1s
    """ 
    ar1s = []
    block_idxs = np.arange(time.shape[0]-win_size, step=offset)
    
    for i in block_idxs:
        try:
            lag0 = arr[i: i+win_size]
            lag1 = arr[i+offset: i+offset+win_size]
            ar1s.append(np.corrcoef(lag0, lag1)[0, 1])
        except:
            print(f'error in {i}')

    return {
        'block_idxs': block_idxs,
        'ar1s': ar1s
    }

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


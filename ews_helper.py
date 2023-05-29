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

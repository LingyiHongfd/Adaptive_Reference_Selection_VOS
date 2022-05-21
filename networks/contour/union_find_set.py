import numpy as np
import numba
from numba import jit


THRESH = 300

@jit(nopython=True, parallel=True)
def init_col(mask, f, tasks):
    h, w = mask.shape
    for j in numba.prange(w):
        for i in range(h-1):
            if mask[i, j] and mask[i+1, j] and f[i, j] != f[i+1, j]:
                a, b = f[i, j], f[i+1, j]
                if a > b:
                    a, b = b, a
                tasks[i, j] = a * h * w + b

@jit(nopython=True,) #parallel=True)
def init_row(mask, id, f, cnt):
    for i in numba.prange(mask.shape[0]):
        w = mask.shape[1]
        j = 0
        while j < w:
            if mask[i,j] == True:
                k = j
                previous = id[i,j]
                while k < mask.shape[1] and mask[i,k] == True:
                    f[i, k] = previous
                    k += 1
                cnt[i, j] = k - j
                j = k
            else:
                j += 1

# def get_group_direct(x, h, w):
@jit(nopython=True, parallel=True)
def calc_groups(f, G):
    h, w = f.shape

    for i in numba.prange(h):
        for j in range(w):
            hash_value = i * w + j

            while hash_value != f[hash_value // w, hash_value % w]:
                hash_value = f[hash_value // w, hash_value % w]
            G[i, j] = hash_value

@jit(nopython=True,)# parallel=True)
def deal_tasks_parallel(unique_tasks, h, w, f, cnt, thresh):
    for each_task in unique_tasks:
        x = each_task // (h * w)
        y = each_task % (h * w)

        while f[x // w, x % w] != x:
            x = f[x // w, x % w]
        while f[y // w, y % w] != y:
            y = f[y // w, y % w]

        if x == y:
            continue

        x_real = x // w, x % w
        y_real = y // w, y % w
        if cnt[x_real] > thresh and cnt[y_real] > thresh:
            continue
        if cnt[x_real] > cnt[y_real]:
            x_real, y_real = y_real, x_real
            x,y = y, x
        cnt[y_real] += cnt[x_real]
        f[x_real] = y


@jit(nopython=True, parallel=True)
def connect(mask, cnt, f, tasks, p):
    h, w = mask.shape
    dh = [-1, 1, 0, 0]
    dw = [0, 0, -1, 1]
    for i in numba.prange(h):
        for j in range(w):
            if not mask[i, j]:
                continue
            i_ = i + dh[p]
            j_ = j + dw[p]
            if i_ < 0 or i_ >=h or j_ < 0 or j_ >= w:
                continue
            u, v = f[i, j], f[i_, j_]
            if u > v:
                u, v = v, u
            tasks[i, j] = h * w * u + v


def segments(edge_pred):
    # global h, w, in_pool, f, cnt, edge, G
    h, w = edge_pred.shape

    id = np.arange(0, h * w).reshape(h, w)  # np.int64
    f = id.copy()  # np.int64
    cnt = np.ones((h, w), np.int64)  # np.int64

    G = id.copy()  # results.... #np.int64

    step = 10.

    for score in range(int(step)):
        mask = (edge_pred >= 1.0 * score / step) * (edge_pred < 1.0 * (score + 1) / step)

        if score == 0:
            init_row(mask, id, f, cnt)
            tasks = np.zeros(mask.shape, np.int64)
            init_col(mask, f, tasks)
            unique_tasks = np.unique(tasks)
            deal_tasks_parallel(unique_tasks, h, w, f, cnt, thresh=300000)
        else:
            tasks_4_dirs = np.zeros((4, h, w), np.int64)
            for p in range(4): #4 directions...
                connect(mask, cnt, f, tasks_4_dirs[p], p)
            unique_tasks = np.unique(tasks_4_dirs)
            deal_tasks_parallel(unique_tasks, h, w, f, cnt, thresh= THRESH)
            calc_groups(f, G)
            f = G

    return G, np.unique(G)



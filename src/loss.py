from src.utils import *


def objective_function(z_x, z_s, z_c, content_weight=4.):

    ix, iy = sample_indices(z_x, z_c)
    x_st, c_st = extract_spatial(z_x, z_c, ix, iy)
    
    s_loss, used_style_feats = remd_loss(x_st, z_s[0])
    c_loss = dp_loss(x_st, c_st)

    m_loss = moment_loss(x_st, z_s[0])
    p_loss, _ = remd_loss(x_st[..., :3], z_s[0][..., :3], dist='l2')

    loss = (content_weight * c_loss + m_loss + s_loss + p_loss / content_weight) \
         / (2. + content_weight + 1/content_weight)
    return loss


def remd_loss(x, y, dist='cos', return_mat=False):

    d = x.shape[1]
    ch = x.shape[-1]

    x = K.reshape(x, (-1, d, ch))
    y = K.reshape(y, (d, ch))

    cx_m = get_d_mat(x, y, dist)
    
    if return_mat:
        return cx_m

    m1, m1_idx = K.min(cx_m, axis=2), K.argmin(cx_m, axis=2)
    m2, m2_idx = K.min(cx_m, axis=1), K.argmin(cx_m, axis=1)
    mean1 = K.mean(m1)
    mean2 = K.mean(m2)

    style_feats = K.switch(mean1 > mean2,
                           K.permute_dimensions(K.gather(y, K.permute_dimensions(m1_idx, (1, 0))), (1, 0, 2)), 
                           y)
    remd = K.mean(K.max([mean1, mean2]))
    
    return remd, style_feats


def dp_loss(x, y):
    d = x.shape[1]
    ch = x.shape[-1]

    x = K.reshape(x, (-1, d, ch))
    y = K.reshape(y, (d, ch))

    m1 = get_d_mat(x, x, 'cos')
    m1 = m1 / K.sum(m1, axis=1, keepdims=True)
    m2 = get_d_mat(y, y, 'cos')
    m2 = m2 / K.sum(m2, axis=1, keepdims=True)

    d = K.mean(K.abs(m1 - m2)) * x.shape[-2]
    return d


def moment_loss(x, y, moments=(1, 2)):
    d = x.shape[1]

    l = 0.

    x = K.reshape(x, (d, -1))
    y = K.reshape(y, (d, -1))

    mu_x = K.mean(x, 0, keepdims=True)
    mu_y = K.mean(y, 0, keepdims=True)
    mu_d = K.mean(K.abs(mu_x - mu_y))

    if 1 in moments:
        l += mu_d

    if 2 in moments:
        d_x, d_y = x - mu_x, y - mu_y

        sig_x = K.transpose(d_x) @ d_x
        sig_y = K.transpose(d_y) @ d_y
        sig_d = K.mean(K.abs(sig_x - sig_y))
        l += sig_d

    return l


def sample_indices(z_x, z_c, n_sample=1024):

    c = 128 ** 2

    ix, iy = [], []

    for x, s in zip(z_x, z_c):
        h, w = x.shape[1:3]
        size = h * w

        stride_x = int(max(np.floor(np.sqrt(size // c)), 1))
        offset_x = np.random.randint(stride_x)

        stride_y = int(max(np.ceil(np.sqrt(size // c)), 1))
        offset_y = np.random.randint(stride_y)

        xx, xy = np.meshgrid(np.arange(h)[offset_x::stride_x], np.arange(w)[offset_y::stride_y])
        xx = xx.flatten()
        xy = xy.flatten()

        r = np.random.permutation(n_sample)
        while np.max(r) > len(xx):
            r = r // 2

        ix.append(xx[r])
        iy.append(xy[r])

    return ix, iy


def extract_spatial(z_x, z_c, xx, xy):
    xs, cs = [], []

    for i, (x, c, ix, iy) in enumerate(zip(z_x, z_c, xx, xy)):
        h, w, d = x.shape[1:]

        idx = ix * x.shape[2] + iy
        x = K.reshape(x, (h * w, d))
        c = K.reshape(c, (h * w, d))

        x = K.reshape(K.gather(x, idx), (1, -1, d))
        c = K.reshape(K.gather(c, idx), (1, -1, d))

        xs.append(x)
        cs.append(c)
    
    x = K.concatenate(xs, axis=-1)
    c = K.concatenate(cs, axis=-1)
    return x, c


def get_d_mat(x, y, dist='cos'):
    if dist == 'cos':
        return pairwise_distance_cos(x, y)
    if dist == 'l2':
        return pairwise_distance_sq_l2(x, y)


def pairwise_distance_cos(x, y):
    d = x.shape[-2]
    x_norm = K.reshape(K.sqrt(K.sum(K.square(x), axis=-1)), (-1, d, 1))
    y_norm = K.reshape(K.sqrt(K.sum(K.square(y), axis=-1)), (-1, 1, d))
    y_t = K.permute_dimensions(y, tuple(range(len(y.shape) - 2)) + (len(y.shape) - 1, len(y.shape) - 2))

    dist = 1. - (x @ y_t) / x_norm / y_norm
    return dist


def pairwise_distance_sq_l2(x, y):
    d = x.shape[-2]
    x_norm = K.reshape(K.sum(K.square(x), axis=-1), (-1, d, 1))
    y_norm = K.reshape(K.sum(K.square(x), axis=-1), (-1, 1, d))
    y_t = K.transpose(y)

    dist = x_norm + y_norm - 2. * K.dot(x, y_t)
    return K.clip(dist, 1e-5, 1e5) / x.shape[1]

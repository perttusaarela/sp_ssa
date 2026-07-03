import numpy as np
from scipy.special import gamma, kv  # kv = modified Bessel function K_nu

def kernel_matrix_from_params(range1, range2, angle):
    """
    Build a single 2x2 SPD kernel matrix from interpretable parameters.

    range1, range2 : local correlation ranges along the two principal axes
    angle          : orientation of axis 1, in radians, measured from x-axis

    Returns the 2x2 matrix Sigma = R diag(range1^2, range2^2) R'
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([range1**2, range2**2])
    return R @ D @ R.T



def gradient_kernel(coords, range_min=0.1, range_max=1.5, angle=0.0, axis=0,
                     domain=(0.0, 1.0)):
    """
    Range grows linearly along one coordinate axis (e.g. smoother / longer-
    range dependence as you move east, or north). Isotropic at each location
    (range1 == range2 == local range), orientation fixed.

    axis   : 0 -> vary with x-coordinate, 1 -> vary with y-coordinate
    domain : (min, max) of that coordinate, used to normalize the gradient
    """
    lo, hi = domain
    t_x = np.clip((coords[:, axis] - lo) / (hi - lo), 0.0, 1.0)
    local_range_x = range_min + t_x * (range_max - range_min)
    t_y = np.clip((coords[:, axis] - lo) / (hi - lo), 0.0, 1.0)
    local_range_y = range_min + t_y * (range_max - range_min)

    n = coords.shape[0]
    Sigmas = np.empty((n, 2, 2))
    for i in range(n):
        Sigmas[i] = kernel_matrix_from_params(local_range_x[i], local_range_y[i], angle)
    return Sigmas


def radial_kernel(coords, center=(0.5, 0.5), a=0.02, b=0.5):
    """
    Range increases (or decreases) radially from a center point -- e.g. tight
    correlation near a coastline/feature at `center`, smoothly relaxing
    outward. Isotropic at each location.

    domain_radius : distance at which range_edge is reached (use roughly the
                    max distance from center to a domain corner)
    """
    center = np.asarray(center)
    d = np.linalg.norm(coords - center, axis=1)
    local_range = a + b * d

    n = coords.shape[0]
    Sigmas = np.empty((n, 2, 2))
    for i in range(n):
        Sigmas[i] = kernel_matrix_from_params(local_range[i], local_range[i], angle=0.0)
    return Sigmas


def custom_function_kernel(coords, range_fn, angle_fn=None, aniso_ratio_fn=None):
    """
    Fully general builder: supply your own functions of (x, y).

    range_fn       : callable, coords (n,2) -> (n,) array of local ranges (axis 1)
    angle_fn       : callable, coords (n,2) -> (n,) array of orientations (radians).
                     Defaults to all zeros (no rotation) if None.
    aniso_ratio_fn : callable, coords (n,2) -> (n,) array of range2/range1 ratios
                     in (0, 1]. Defaults to all ones (isotropic) if None.

    Example:
        range_fn = lambda c: 0.1 + 0.4*np.sin(2*np.pi*c[:,0])**2
    """
    n = coords.shape[0]
    r1 = range_fn(coords)
    angle = angle_fn(coords) if angle_fn is not None else np.zeros(n)
    ratio = aniso_ratio_fn(coords) if aniso_ratio_fn is not None else np.ones(n)
    r2 = r1 * ratio

    Sigmas = np.empty((n, 2, 2))
    for i in range(n):
        Sigmas[i] = kernel_matrix_from_params(r1[i], r2[i], angle[i])
    return Sigmas


# ----------------------------------------------------------------------
# 2. Generic machinery (you shouldn't need to edit below this line)
# ----------------------------------------------------------------------

def matern_correlation(h, nu=1.5):
    """
    Stationary isotropic Matern correlation at (already scaled) distance h>=0.
    h is dimensionless here -- the P-S formula folds all the local scaling
    into the quadratic form Q_ij before this function is ever called, so this
    is just the base shape function K_base(sqrt(Q_ij)).
    """
    h = np.asarray(h, dtype=float)
    out = np.ones_like(h)
    mask = h > 1e-12
    hm = h[mask]
    # Matern formula; special-cased for numerically common nu values for speed/stability
    if np.isclose(nu, 0.5):
        out[mask] = np.exp(-hm)
    elif np.isclose(nu, 1.5):
        out[mask] = (1 + hm) * np.exp(-hm)
    elif np.isclose(nu, 2.5):
        out[mask] = (1 + hm + hm**2 / 3) * np.exp(-hm)
    else:
        const = 2 ** (1 - nu) / gamma(nu)
        out[mask] = const * (hm ** nu) * kv(nu, hm)
    return out


def to_unit_square(coords, domain_x=None, domain_y=None):
    """
    Linearly rescale coords from a rectangular domain to [0,1] x [0,1].

    coords    : (n, 2) array of locations
    domain_x  : (xmin, xmax) of the original domain. If None, inferred from
                coords (i.e. min/max of the x-column) -- convenient, but
                only correct if your points actually span the full domain.
    domain_y  : (ymin, ymax) of the original domain. If None, inferred from
                coords. Pass domain_x=(0, ell), domain_y=(0, ell) explicitly
                for the common case of a square domain [0, ell] x [0, ell],
                especially if your sampled locations don't reach the edges.

    Returns (unit_coords, domain_x, domain_y) -- the rescaled coordinates
    plus the domain actually used.
    """
    coords = np.asarray(coords, dtype=float)
    if domain_x is None:
        domain_x = (coords[:, 0].min(), coords[:, 0].max())
    if domain_y is None:
        domain_y = (coords[:, 1].min(), coords[:, 1].max())

    xlo, xhi = domain_x
    ylo, yhi = domain_y
    if xhi <= xlo or yhi <= ylo:
        raise ValueError("domain_x/domain_y must have max > min.")

    unit_coords = np.empty_like(coords)
    unit_coords[:, 0] = (coords[:, 0] - xlo) / (xhi - xlo)
    unit_coords[:, 1] = (coords[:, 1] - ylo) / (yhi - ylo)
    return unit_coords, domain_x, domain_y


def from_unit_square(unit_coords, domain_x, domain_y):
    """Inverse of to_unit_square: map [0,1] x [0,1] coordinates back to the
    original (domain_x, domain_y) rectangle."""
    unit_coords = np.asarray(unit_coords, dtype=float)
    xlo, xhi = domain_x
    ylo, yhi = domain_y
    coords = np.empty_like(unit_coords)
    coords[:, 0] = unit_coords[:, 0] * (xhi - xlo) + xlo
    coords[:, 1] = unit_coords[:, 1] * (yhi - ylo) + ylo
    return coords


def build_ps_covariance(coords, local_kernel_fn, sigma2=1.0, nu=1.5,
                         nugget=1e-6, kernel_kwargs=None,
                        domain_x=None, domain_y=None, rescale=True):
    """
    Build the full n x n Paciorek-Schervish nonstationary covariance matrix.

    coords          : (n, 2) array of locations
    local_kernel_fn : function coords -> (n, 2, 2) array of local SPD kernels
                       (one of the builders above, or your own)
    sigma2          : overall variance (sill)
    nu              : Matern smoothness of the base correlation shape
    nugget          : small value added to the diagonal for numerical
                       stability (and to represent measurement-error-free
                       "microscale" variation if you want some)
    kernel_kwargs   : dict of extra keyword args passed to local_kernel_fn

    Returns (C, Sigmas) where C is the (n, n) covariance matrix and Sigmas is
    the (n, 2, 2) array of local kernels actually used.
    """
    kernel_kwargs = kernel_kwargs or {}
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    if rescale:
        coords, domain_x, domain_y = to_unit_square(coords, domain_x, domain_y)

    Sigmas = local_kernel_fn(coords, **kernel_kwargs)  # (n, 2, 2)

    a = Sigmas[:, 0, 0]   # (n,)
    b = Sigmas[:, 0, 1]
    d = Sigmas[:, 1, 1]

    dets = a * d - b * b                        # (n,) det(Sigma_i)
    if np.any(dets <= 0):
        raise ValueError("Local kernel matrices must be positive definite")
    det_pow = dets ** 0.25                       # (n,), this is |Sigma_i|^(1/4)

    dx = coords[:, 0][:, None] - coords[:, 0][None, :]   # (n,n)
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]   # (n,n)

    # Sigma_avg entries, pairwise: 0.5*(Sigma_i + Sigma_j)
    A = 0.5 * (a[:, None] + a[None, :])          # (n,n)
    B = 0.5 * (b[:, None] + b[None, :])          # (n,n)
    D = 0.5 * (d[:, None] + d[None, :])          # (n,n)

    det_avg = A * D - B * B                      # (n,n) det(Sigma_avg)
    # Q_ij = diff' * inv(Sigma_avg) * diff, using the closed-form 2x2 inverse:
    # inv([[A,B],[B,D]]) = (1/det) * [[D,-B],[-B,A]]
    Q = (D * dx * dx - 2 * B * dx * dy + A * dy * dy) / det_avg
    Q = np.maximum(Q, 0.0)                       # guard tiny negatives

    prefactor = (det_pow[:, None] * det_pow[None, :]) / np.sqrt(det_avg)
    C = sigma2 * prefactor * matern_correlation(np.sqrt(Q), nu=nu)

    C[np.diag_indices(n)] += nugget
    return C, Sigmas


def simulate_gp(C, n_realizations=1, mean=0.0, random_state=None,
                 jitter=1e-8, max_jitter_tries=6):
    """
    Simulate Gaussian random field realizations from covariance C via
    Cholesky decomposition. If C is not quite numerically PD (common with
    nonstationary covariances near machine precision), retries with
    increasing diagonal jitter.

    Returns an (n_realizations, n) array.
    """
    rng = np.random.default_rng(random_state)
    n = C.shape[0]
    C_try = C.copy()
    add = 0.0
    for attempt in range(max_jitter_tries):
        try:
            L = np.linalg.cholesky(C_try)
            break
        except np.linalg.LinAlgError:
            add = jitter if add == 0.0 else add * 10
            C_try = C + add * np.eye(n)
            if attempt == max_jitter_tries - 1:
                raise
    z = rng.standard_normal((n, n_realizations))
    samples = (L @ z).T + mean
    return samples


def get_nonstationary_signals(coords):
    N_LOCATIONS = coords.shape[0]
    l = np.sqrt(N_LOCATIONS)
    DOMAIN = (0.0, l)  # square domain [0,1] x [0,1]

    scenarios = [
        ("Gradient",
         gradient_kernel,
         dict(range_min=0.05, range_max=1.35, angle=np.pi/12, axis=0, domain=DOMAIN)),

        ("Radial",
         radial_kernel,
         dict(center=(0.5, 0.5), a=0.02, b=0.5)),

        ("sinusoidal range field",
         custom_function_kernel,
         dict(range_fn=lambda c: 0.05 + 0.25 * np.sin(2 * np.pi * c[:, 0]) ** 2,
              angle_fn=lambda c: np.pi * c[:, 1],
              aniso_ratio_fn=lambda c: np.full(c.shape[0], 0.5))),
    ]

    SIGMA2 = 1.0  # overall variance, shared across scenarios
    NU = 1.5  # Matern smoothness, shared across scenarios
    NUGGET = 1e-6

    samples = []
    for j, (label, kfn, kwargs) in enumerate(scenarios):
        C, Sigmas = build_ps_covariance(coords, kfn, sigma2=SIGMA2, nu=NU,
                                        nugget=NUGGET, kernel_kwargs=kwargs)
        sample = simulate_gp(C, n_realizations=1)[0]
        samples.append(sample)

    ret = np.empty((3, N_LOCATIONS))

    ret[0] = samples[0]
    ret[1] = samples[1]
    ret[2] = samples[2]

    ret -= ret.mean(axis=1, keepdims=True)
    ret /= ret.std(axis=1, keepdims=True)

    return ret
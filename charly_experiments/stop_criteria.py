import jax.numpy as jnp


def partial_pearson(fixed):
    """Binds 'fixed' and returns
    a function that calculates
    pearson correlation, unsigned.
    Parameters
    """
    l = fixed.shape[0]
    am = fixed - jnp.sum(fixed, axis=0) / l
    aa = jnp.sum(am ** 2, axis=0) ** 0.5

    def get(b):
        bm = b - jnp.sum(b, axis=0) / l
        bb = jnp.sum(bm ** 2, axis=0) ** 0.5
        ab = jnp.sum(am * bm, axis=0)
        return jnp.abs(ab / (aa * bb))

    return get


array_1 = jnp.array([1., 2., 3., 4.])
array_2 = jnp.array([[1., 2.], [3., 4.]])
pp = partial_pearson(jnp.array([1., 2., 3., 4.]))
print(pp(jnp.array([1., 2., 3., 4.])))
print(pp(jnp.array([-1., -2., -3., -4.])))

pp = partial_pearson(array_2)
print(pp(jnp.array([[1., 2.], [3., 4.]])))


# TODO HOW TO APPLY TO MULTIVARIABLE?

def estatistic(x):
    return jnp.add(x, jnp.power(x, 2))


print(estatistic(array_1))
print(estatistic(array_2))


def build_criteria1(init_particles, alpha, t):
    """

    Parameters
    ----------
    init_particles
    alpha
    t

    Returns
    -------

    """
    pp = partial_pearson(init_particles)
    def criteria1(prev, new):
        return jnp.mean(pp(prev)-pp(new) > alpha) > t

    return criteria1

stop_criteria = build_criteria1(array_1, 50, 10)
print(stop_criteria(jnp.add(array_1, jnp.array([4,4,2,4])),
                    jnp.add(array_1, jnp.array([2,2,0,10]))))


stop_criteria = build_criteria1(array_2, 50, 10)
print(stop_criteria(jnp.add(array_2, jnp.array([[4,4],[2,4]])),
                    jnp.add(array_2, jnp.array([[2,2],[0,10]))))




def build_criteria2(init_particles, alpha, t):
    """

    Parameters
    ----------
    init_particles
    alpha
    t

    Returns
    -------

    """
    pp = partial_pearson(init_particles)
    def criteria1(prev, new):
        return jnp.mean(pp(prev)-pp(new) > alpha) > t

    return criteria1
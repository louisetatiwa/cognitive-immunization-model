def credit_assignment(m_g, s2_g, s2_d, sigma2, y):
    """Split one prediction error between the general belief g and the task belief d.

    Model: y = g + d + noise, with
        g ~ N(m_g, s2_g)    general self-belief
        d ~ N(0, s2_d)      task-specific deviation
        noise ~ N(0, sigma2)
    g, d and noise are independent before the observation.

    Parameters
    ----------
    m_g : float     prior mean of g
    s2_g : float    prior variance of g
    s2_d : float    prior variance of d (prior mean of d is 0)
    sigma2 : float  feedback noise variance
    y : float       observed feedback

    Returns
    -------
    dict with keys:
        "y_pred"      predicted feedback before observing y
        "e"           prediction error
        "K_g", "K_d"  weights of the prediction error for g and d
        "m_g_new"     posterior mean of g
        "m_d_new"     posterior mean of d
        "s2_g_new"    posterior variance of g
        "s2_d_new"    posterior variance of d
        "cov_gd_new"  posterior covariance between g and d
    """
    # Exercise 1: predicted feedback and its variance
    y_pred = m_g                    # E[y] = m_g + 0 + 0
    s2_y = s2_g + s2_d + sigma2     # Var[y] = sum of independent variances

    # Prediction error
    e = y - y_pred

    # Exercise 2: weight = Cov(x, y) / Var(y), with Cov(g, y) = s2_g and Cov(d, y) = s2_d
    K_g = s2_g / s2_y
    K_d = s2_d / s2_y

    # Exercise 2: new means
    m_g_new = m_g + K_g * e
    m_d_new = 0 + K_d * e

    # Conditional variance: Var(x) - Cov(x, y)^2 / Var(y)
    s2_g_new = s2_g - s2_g**2 / s2_y
    s2_d_new = s2_d - s2_d**2 / s2_y

    # Exercise 5: new covariance = 0 - Cov(g, y) * Cov(d, y) / Var(y)
    cov_gd_new = 0 - (s2_g * s2_d) / s2_y

    return {
        "y_pred": y_pred,
        "e": e,
        "K_g": K_g,
        "K_d": K_d,
        "m_g_new": m_g_new,
        "m_d_new": m_d_new,
        "s2_g_new": s2_g_new,
        "s2_d_new": s2_d_new,
        "cov_gd_new": cov_gd_new,
    }
import pytest
from credit_assignment import credit_assignment
from bayes import update_gaussian  # AJUSTAR: nome do arquivo da Phase 1


# Hand-calculated example: s2_g = 4, s2_d = 1, sigma2 = 5 -> Var(y) = 10
PARAMS = dict(m_g=0, s2_g=4, s2_d=1, sigma2=5, y=1)


def test_prediction_and_error():
    r = credit_assignment(**PARAMS)
    assert r["y_pred"] == pytest.approx(0)
    assert r["e"] == pytest.approx(1)


def test_weights():
    r = credit_assignment(**PARAMS)
    assert r["K_g"] == pytest.approx(0.4)   # 4 / 10
    assert r["K_d"] == pytest.approx(0.1)   # 1 / 10


def test_new_means():
    r = credit_assignment(**PARAMS)
    assert r["m_g_new"] == pytest.approx(0.4)
    assert r["m_d_new"] == pytest.approx(0.1)


def test_new_means_negative_surprise():
    # m_g = 2, y = 1 -> e = -1: both beliefs move down
    r = credit_assignment(m_g=2, s2_g=4, s2_d=1, sigma2=5, y=1)
    assert r["m_g_new"] == pytest.approx(1.6)   # 2 + 0.4 * (-1)
    assert r["m_d_new"] == pytest.approx(-0.1)  # 0 + 0.1 * (-1)


def test_new_variances():
    r = credit_assignment(**PARAMS)
    assert r["s2_g_new"] == pytest.approx(2.4)  # 4 - 16/10
    assert r["s2_d_new"] == pytest.approx(0.9)  # 1 - 1/10


def test_new_covariance_is_negative():
    r = credit_assignment(**PARAMS)
    assert r["cov_gd_new"] == pytest.approx(-0.4)  # -(4 * 1) / 10
    assert r["cov_gd_new"] < 0


def test_symmetry():
    # equal uncertainty at both levels -> both means move by the same amount
    r = credit_assignment(m_g=0, s2_g=2, s2_d=2, sigma2=3, y=1)
    assert r["m_g_new"] == pytest.approx(r["m_d_new"])


def test_consistent_with_phase_1():
    # the two levels together behave like one level with s2 = s2_g + s2_d
    r = credit_assignment(**PARAMS)
    m_new, _ = update_gaussian(m=0, s2=4 + 1, y=1, sigma2=5)
    assert r["m_g_new"] + r["m_d_new"] == pytest.approx(m_new)


def test_split_does_not_depend_on_sigma2():
    # exercise 3: share for g = s2_g / (s2_g + s2_d), whatever sigma2 is
    r_low = credit_assignment(m_g=0, s2_g=4, s2_d=1, sigma2=0.1, y=1)
    r_high = credit_assignment(m_g=0, s2_g=4, s2_d=1, sigma2=100, y=1)
    share_low = r_low["K_g"] / (r_low["K_g"] + r_low["K_d"])
    share_high = r_high["K_g"] / (r_high["K_g"] + r_high["K_d"])
    assert share_low == pytest.approx(0.8)   # 4 / (4 + 1)
    assert share_high == pytest.approx(0.8)
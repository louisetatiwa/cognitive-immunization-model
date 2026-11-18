from bayes import update_gaussian
import pytest


def test_equal_variance():
    m=3
    s2=2
    y=6
    sigma2=2
    m_new, s2_new = update_gaussian(m,s2,y,sigma2)
    assert m_new == 0.5*(y+m)
          
    

def test_hand_example():
    m = 3
    s2= 1
    y = 5
    sigma2 = 1
    m_new, s2_new = update_gaussian(m,s2,y,sigma2)
    assert m_new == 4
    assert s2_new == pytest.approx(0.5)
    
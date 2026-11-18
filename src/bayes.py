def update_gaussian(m,s2,y,sigma2):
    K= s2/(s2+sigma2)
    m_new= m+K*(y-m)
    s2_new= (1-K)*s2
    return m_new, s2_new

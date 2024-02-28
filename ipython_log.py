# IPython log file

get_ipython().run_cell_magic("quickref", "", "")
get_ipython().magic("quickref")
b = [1, 2, 3]
get_ipython().magic("pinfo b")
b = [1, 2, 3]
get_ipython().magic("pinfo b")
get_ipython().magic("pinfo str.split")
get_ipython().magic("pinfo2 str.split")


def gcd(a, b):
    """Computes the greatest common divisor of integers a and b using
    Euclid's Algorithm.
    """
    while True:
        if b == 0:
            return a
        a, b = b, a % b


get_ipython().magic("pinfo gcd")
import uuid

get_ipython().magic("pinfo2 uuid.uuid4")
get_ipython().magic("pinfo str.split")
get_ipython().magic("psearch str.*split*")
get_ipython().magic("run gcd.py")
print(a, b)
get_ipython().magic("run gcd.py")
import numpy as np

a = np.random.randn(100, 100)
a = np.random.randn(100, 100)
get_ipython().magic("timeit np.dot(a, a)")
get_ipython().magic("pinfo %reset")
"a" in _ip.user_ns
get_ipython().magic("magic")
print(_)
get_ipython().magic("run gcd.py")
print(_)
get_ipython().magic("run gcd.py")
print(_)
get_ipython().magic("run gcd.py")
get_ipython().magic("run gcd.py")
366 * 31 * 24 * 60 * 60
print(_)
_ * 10
366 * 31 * 24 * 60 * 60
_ * 2
print(_, __)
two_years_sec = 980294400
print(_i34, _34)
two_years_sec = 980294400
two_years_sec
print(_i36, _36)
print(_i36)
print(_i36)
print(_36)
exec(_i36)
get_ipython().magic("hist")
get_ipython().magic("logstart")
get_ipython().magic("logoff")

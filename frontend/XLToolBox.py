import sys

module_path = r"F:\NewJersey\XWei\ResQ"

if module_path not in sys.path:
    sys.path.insert(0, module_path)


from XLToolBox2 import *
from ProjectDateTime import *

def test_func(arg1='ag1', arg2=[1,2,3]):
    print(arg1)
    print(arg2.append(5))

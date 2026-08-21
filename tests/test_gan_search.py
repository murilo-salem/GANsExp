from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import numpy as np

spec=spec_from_file_location('search',Path(__file__).parents[1]/'code/pipeline/stage11_gan_search.py'); mod=module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)

def test_roll_zero_does_not_wrap_pixels():
    x=np.ones((4,4,1),np.float32); y=mod.roll_zero(x,1,0)
    assert y[0].sum()==0 and y[1:].sum()==12

def test_phase_shift_is_bounded():
    x=np.zeros((32,32,3),np.float32); x[10:15,10:15,2]=1; y=mod.roll_zero(x,3,-2)
    dy,dx,_=mod.shift_phase(x,y,np.ones((32,32),np.float32))
    assert abs(dy)<=8 and abs(dx)<=8

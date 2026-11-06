import numpy as np

from milho_experiment.pipeline.stage_05_synthesis import stage11_gan_search as mod

def test_roll_zero_does_not_wrap_pixels():
    x=np.ones((4,4,1),np.float32); y=mod.roll_zero(x,1,0)
    assert y[0].sum()==0 and y[1:].sum()==12

def test_phase_shift_is_bounded():
    x=np.zeros((32,32,3),np.float32); x[10:15,10:15,2]=1; y=mod.roll_zero(x,3,-2)
    dy,dx,_=mod.shift_phase(x,y,np.ones((32,32),np.float32))
    assert abs(dy)<=8 and abs(dx)<=8

def test_architecture_registry_separates_inputs_and_disables_wgangp_by_default():
    assert mod.CONFIGS['identity']['input_stages']==['R2']
    assert mod.CONFIGS['convlstm_lite']['input_stages']==['V8','R2']
    assert mod.CONFIGS['wgangp']['enabled_by_default'] is False

# MW Irvin -- Lopez Lab -- 2018-10-01
"""
========================================
Simple PySB Model of Apoptosis Signaling
========================================

PySB Model of the following Apoptosis signaling reactions:

1. Receptor ligation:                   TRAIL + Receptor <-> TRAIL:Receptor
2. Death inducing signaling complex:    TRAIL:Receptor   --> DISC
3. Caspase Activation: DISC->                  Caspases  --> *Caspases
4. Caspase Activation: *Caspases->             Caspases  --> *Caspases
5. *Caspases->  PARP      --> cPARP
"""

from pysb import *
from pysb.macros import catalyze_state
Model()

Parameter('L_0',    3000)  # baseline level of TRAIL in most experiments (50 ng/ml SuperKiller TRAIL)
Parameter('R_0',    200)   # TRAIL receptor (for experiments not involving siRNA)
Parameter('C_0',    1e4)   # Caspases
Parameter('DISC_0',   0)     # Death inducing signaling complex
Parameter('PARP_0', 1e6)   # PARP (Capase-3 substrate)


Monomer('L', ['b'])
Monomer('R', ['b'])
Monomer('C', ['b', 'state'], {'state': ['inactive', 'active']})
Monomer('DISC', ['b'])
Monomer('PARP', ['b', 'state'], {'state': ['unmod', 'cleaved']})

Initial(L(b=None), L_0)
Initial(R(b=None), R_0)
Initial(C(b=None, state='inactive'), C_0)
Initial(DISC(b=None), DISC_0)
Initial(PARP(b=None, state='unmod'), PARP_0)

Parameter('kf0', 1e-06)
Parameter('kr0', 1e-03)
Rule('Receptor_ligation', L(b=None) + R(b=None) | L(b=1) % R(b=1), kf0, kr0)

Parameter('kc1', 1e-06)
Rule('DISC_formation', L(b=1) % R(b=1) >> DISC(b=None), kc1)

Parameter('kf2', 1e-06)
Parameter('kr2', 1e-03)
Parameter('kc2', 1e+00)
catalyze_state(DISC(), 'b', C(), 'b', 'state', 'inactive', 'active', [kf2, kr2, kc2])

Parameter('kf3', 1e-06)
Parameter('kr3', 1e-03)
Parameter('kc3', 1e+00)
catalyze_state(C(state='active'), 'b', C(), 'b', 'state', 'inactive', 'active', [kf3, kr3, kc3])

Parameter('kf4', 1e-06)
Parameter('kr4', 1e-03)
Parameter('kc4', 1e+00)
catalyze_state(C(state='active'), 'b', PARP(), 'b', 'state', 'unmod', 'cleaved', [kf4, kr4, kc4])

Observable('cPARP_obs', PARP(b=None, state='cleaved'))
Observable('PARP_obs',  PARP(b=None, state='unmod'))
# Observable('Caspase_obs', C(state='active'))




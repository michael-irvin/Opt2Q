# MW Irvin -- Lopez Lab -- 2018-12-19

"""
=========================
Logistic Classifier Model
=========================

Nominal observations provide information about the quantifiable attributes (i.e. markers) on which they depend.

For example, programmed cell death (apoptosis) depends on caspase activity; as such, apoptotic cells will more likely
have similar caspase activity that differs from that of surviving cells.

`Albeck and Sorger 2015 <http://msb.embopress.org/content/11/5/803.long>`_ find that the maximum rate of change in
caspase indicator, and the time when that maximum occurs, predicts cellular commitment to apoptosis with 83% accuracy.

The following uses the :class:`~opt2q.measurement.FractionalKilling` to calibrate a model of apoptosis to cell viability
measurements by Albeck and Sorger.
"""


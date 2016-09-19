import sys, pickle, matplotlib
import matplotlib.pyplot as pl
from cycler import cycler

pgf_with_rc_fonts = {
    "font.family": "Arial",
    "font.serif": [],
    "font.sans-serif": ["Arial"],
    "font.size": 10,
    "axes.prop_cycle": (cycler('color', [
            (0, 166./255, 214./255),
            (165./255, 202./255, 26./255),
            (226./255, 26./255, 26./255),
            (109./255, 23./255, 127./255)
        ]))
}
matplotlib.rcParams.update(pgf_with_rc_fonts)

name = sys.argv[1]

p = pickle.load(open("results/{}-comp.p".format(name), "rb"))

pl.figure(figsize=(4, 2))

pl.plot(p["compliances"])

ax = pl.gca()

ax.set_xlabel('iteration')
ax.set_ylabel('compliance')

pl.savefig("results/{}-compliances.pdf".format(name), bbox_inches="tight")
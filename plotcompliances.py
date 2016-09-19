import sys, pickle, matplotlib
import matplotlib.pyplot as pl
from cycler import cycler

pgf_with_rc_fonts = {
    "font.family": "Arial",
    "font.serif": [],
    "font.sans-serif": ["Arial"],
    "font.size": 8,
    "figure.dpi": 600,
    "axes.prop_cycle": (cycler('color', [
            (0, 166./255, 214./255),
            (165./255, 202./255, 26./255),
            (226./255, 26./255, 26./255),
            (109./255, 23./255, 127./255)
        ]))
}
matplotlib.rcParams.update(pgf_with_rc_fonts)

infile = sys.argv[1]
outfile = sys.argv[2]

p = pickle.load(open(infile, "rb"))

pl.figure(figsize=(3, 1.5))

if len(sys.argv) == 4 and sys.argv[3] == "z":
  start = 400
  comp = p["compliances"][start:]
  x = range(start, start + len(comp))

  pl.plot(x, comp)
else:
  pl.plot(p["compliances"])

ax = pl.gca()

ax.set_xlabel('iteration')
ax.set_ylabel('compliance')

pl.savefig(outfile, bbox_inches="tight", dpi="figure")

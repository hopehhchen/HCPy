from matplotlib import rcParams
from cycler import cycler

colors_538 = ["#30a2da",
              "#fc4f30",
              "#e5ae38",
              "#6d904f",
              "#8b8b8b"]

### Hope's implementation ###
# lines
rcParams['lines.linewidth'] = 1
rcParams['lines.markersize'] = 20
# image
rcParams['image.cmap'] = 'viridis'
rcParams['image.interpolation'] = 'none'
rcParams['image.origin'] = 'bottom'
# patch
rcParams['patch.facecolor'] = 'yellow'
rcParams['patch.edgecolor'] = 'none'
# font
rcParams['font.size'] = 26
rcParams['font.family'] = 'StixGeneral'
# legend
rcParams['legend.frameon'] = False
rcParams['legend.scatterpoints'] = 1
rcParams['legend.numpoints'] = 1
rcParams['legend.fontsize'] = 26
# axes
rcParams['axes.prop_cycle'] = cycler('color', colors_538)
rcParams['axes.facecolor'] = 'none'
# figure
rcParams['figure.figsize'] = (14, 14)
rcParams['figure.dpi'] = 180
rcParams['figure.subplot.left'] = .1
rcParams['figure.subplot.right'] = .97
rcParams['figure.subplot.bottom'] = .1
rcParams['figure.subplot.top'] = .96
rcParams['figure.subplot.wspace'] = .02
rcParams['figure.subplot.hspace'] = .02
# savefig
rcParams['savefig.jpeg_quality'] = 100
rcParams['savefig.dpi'] = 180
rcParams['savefig.directory'] = '/Users/hopechen/Desktop/'

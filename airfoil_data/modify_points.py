import geopandas as gpd
from shapely import Polygon, affinity
import numpy as np

file = "wing_formatted.dat"

vertices = np.genfromtxt(file, skip_header = 2, usecols = (0, 1))

polygon = Polygon(vertices)
buffered_polygon = polygon.buffer(0.00001, join_style=2)
# gdf = gpd.GeoDataFrame({'geometry': [buffered_polygon, polygon]})
# gdf.plot(column='geometry')

xx, yy = buffered_polygon.exterior.coords.xy
f = open("airfoil_formatted_buffered.dat", "w")
f.write("CRVS\n")
f.write(f"{len(xx)}\n")
for i in range(len(xx)):
    f.write(f"{xx[i]} {yy[i]} {0}\n")
f.close()
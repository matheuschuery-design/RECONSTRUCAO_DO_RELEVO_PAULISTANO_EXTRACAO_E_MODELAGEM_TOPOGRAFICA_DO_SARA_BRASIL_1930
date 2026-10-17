from qgis.core import QgsProject, QgsVectorLayer, QgsApplication
import processing, os

layer = iface.activeLayer()
inp_path = layer.dataProvider().dataSourceUri().split('|')[0]
pasta = os.path.dirname(inp_path)
base = os.path.splitext(os.path.basename(inp_path))[0]
out = os.path.join(pasta, f"{base}_conectada.gpkg")

# 1) break
p1 = os.path.join(pasta, "_tmp_break.gpkg")
processing.run("grass:v.clean", {
    'input': inp_path, 'type': [0,1,2,3,4,5,6],
    'tool': 0, 'threshold': [0.0], 'error': p1 + ".err",
    'output': p1})

# 2) snap 20m
p2 = os.path.join(pasta, "_tmp_snap.gpkg")
processing.run("grass:v.clean", {
    'input': p1, 'type': [0,1,2,3,4,5,6],
    'tool': 1, 'threshold': [20.0], 'error': p2 + ".err",
    'output': p2})

# 3) rmdangle 30m
p3 = os.path.join(pasta, "_tmp_dangle.gpkg")
processing.run("grass:v.clean", {
    'input': p2, 'type': [0,1,2,3,4,5,6],
    'tool': 2, 'threshold': [30.0], 'error': p3 + ".err",
    'output': p3})

# 4) smooth
processing.run("native:smooth", {
    'INPUT': p3, 'ITERATIONS': 2, 'OFFSET': 0.25,
    'MIN_DISTANCE': 0.0, 'OUTPUT': out})

# adiciona no QGIS
lyr = QgsVectorLayer(out, "curvas_conectadas", "ogr")
if lyr.isValid():
    QgsProject.instance().addMapLayer(lyr)
    print(f"✅ {out}")
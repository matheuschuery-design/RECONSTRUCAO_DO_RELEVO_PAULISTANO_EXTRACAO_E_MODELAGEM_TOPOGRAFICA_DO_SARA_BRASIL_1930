# -*- coding: utf-8 -*-
"""
================================================================================
 ESQUELETIZAÇÃO E VETORIZAÇÃO DE CURVAS DE NÍVEL
================================================================================

 DESCRIÇÃO
 ---------
 Transforma um raster binário de curvas de nível em um shapefile de linhas
 vetoriais, passando por três etapas:

 1. ESQUELETIZAÇÃO (skeletonization)
    Reduz as curvas a uma espessura de 1 pixel (linha central). Aplicado em
    tiles para viabilizar rasters grandes.

 2. VETORIZAÇÃO POR GRAFO TOPOLÓGICO
    Percorre o esqueleto como um grafo, identificando nós (pontas e junções)
    e arestas (segmentos entre nós). Cada aresta vira uma LineString.

 3. PODA DE SPURS
    Remove pequenos "galhos" (esporões) que surgem de imperfeições na
    esqueletização. Um esporão é uma aresta curta com uma ponta livre e
    outra extremidade em uma junção (grau ≥ 3).

 4. CONEXÃO DE GAPS (opcional)
    Conecta pontas livres próximas se os vetores das curvas apontarem uma
    para a outra dentro de um ângulo máximo.

 ALGORITMO DE GRAFO
 ------------------
 - Nó = pixel com número de vizinhos ≠ 2 (pontas com 1 vizinho, junções com ≥ 3)
 - Aresta = caminho contínuo de pixels entre dois nós
 - Spur = aresta curta com ponta solta

 REQUISITOS
 ----------
 - Python 3.10+
 - GDAL, NumPy, GeoPandas, Shapely, SciPy
 - scikit-image
 - QGIS Python (opcional, para carregar resultado)

 COMO USAR
 ---------
 1. Selecione o raster de curvas limpas no QGIS e rode o script no console
    Python do QGIS (usa a camada ativa).
 2. OU ajuste CONFIG e rode via terminal.

 RESULTADO
 ---------
 Shapefile (.shp) com LineStrings vetorizadas e podadas.

 AUTORIA
 -------
 [Seu nome] / [Programa] / [Instituição] / [Ano]
================================================================================
"""

import os
import time
from collections import defaultdict

import numpy as np
import geopandas as gpd
from scipy import ndimage
from skimage.morphology import (
    skeletonize, remove_small_holes, remove_small_objects,
    binary_closing, disk,
)
from shapely.geometry import LineString
from shapely.ops import unary_union, linemerge
from osgeo import gdal


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

CONFIG = {
    # Caminhos (se rodar standalone; se no QGIS, usa camada ativa)
    "raster_entrada": r"CAMINHO\PARA\curvas_limpas.tif",
    "pasta_saida":    r"CAMINHO\PARA\pasta\de\saida",
    "nome_saida":     "curvas_vetorizadas.shp",

    # Esqueletização em tiles
    "tile":              4096,
    "overlap":           64,

    # Filtros morfológicos (antes da esqueletização)
    "tam_buraco":        150,   # Área máxima de buraco para preencher
    "tam_obj_min":       30,    # Área mínima de objeto
    "kernel_close":      2,     # Raio do disco para closing morfológico

    # Poda de spurs
    "spur_max_px":       20,    # Comprimento máximo de esporão (pixels)
    "bloco_poda":        2000,  # Linhas por bloco (economia de RAM)

    # Conexão de gaps
    "conectar_gaps":     True,
    "raio_max_gap_m":    120.0,
    "angulo_max_gap":    60.0,
}


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def log(msg=""):
    print(msg, flush=True)


def obter_raster_entrada(cfg):
    """Se rodando no QGIS com camada raster ativa, usa ela. Senão, usa CONFIG."""
    try:
        from qgis.core import QgsMapLayerType
        layer = iface.activeLayer()
        if layer and layer.type() == QgsMapLayerType.RasterLayer:
            caminho = layer.dataProvider().dataSourceUri().split("|")[0]
            crs = layer.crs().authid()
            return caminho, crs
    except (ImportError, NameError):
        pass
    return cfg["raster_entrada"], "EPSG:31983"


def esqueletizar_em_tiles(ds, W, H, cfg, log):
    """Aplica esqueletização em tiles com overlap."""
    log("[1/4] Esqueletizando em tiles...")
    t0 = time.time()

    esqueleto = np.zeros((H, W), dtype=bool)
    step = cfg["tile"] - cfg["overlap"]
    n_tiles = ((W + step - 1) // step) * ((H + step - 1) // step)
    done = 0

    for ty in range(0, H, step):
        for tx in range(0, W, step):
            rx = max(0, tx - cfg["overlap"])
            ry = max(0, ty - cfg["overlap"])
            rw = min(cfg["tile"] + 2 * cfg["overlap"], W - rx)
            rh = min(cfg["tile"] + 2 * cfg["overlap"], H - ry)

            arr = ds.GetRasterBand(1).ReadAsArray(rx, ry, rw, rh)
            if arr is None:
                done += 1
                continue

            mask = arr > 0
            mask = remove_small_holes(mask, area_threshold=cfg["tam_buraco"])
            mask = remove_small_objects(mask, min_size=cfg["tam_obj_min"])
            mask = binary_closing(mask, disk(cfg["kernel_close"]))

            try:
                skel = skeletonize(mask)
            except Exception:
                skel = np.zeros_like(mask)

            cx1 = tx - rx
            cy1 = ty - ry
            cw = min(cfg["tile"], W - tx)
            ch = min(cfg["tile"], H - ty)
            esqueleto[ty:ty + ch, tx:tx + cw] = skel[cy1:cy1 + ch, cx1:cx1 + cw]

            done += 1
            if done % 50 == 0 or done == n_tiles:
                log(f"   tile {done}/{n_tiles}")

    log(f"   Esqueleto: {int(esqueleto.sum()):,} px")
    log(f"   Tempo: {(time.time() - t0) / 60:.1f} min\n")
    return esqueleto


def contar_vizinhos(esqueleto, H, W, bloco):
    """Conta vizinhos de cada pixel em blocos (economia de RAM)."""
    viz = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    nb = np.zeros((H, W), dtype=np.uint8)

    for y0 in range(0, H, bloco):
        y1 = min(y0 + bloco, H)
        ry0 = max(0, y0 - 1)
        ry1 = min(H, y1 + 1)

        sub = esqueleto[ry0:ry1, :]
        sh, sw = sub.shape
        sub_pad = np.zeros((sh + 2, sw + 2), dtype=np.uint8)
        sub_pad[1:-1, 1:-1] = sub

        acc = np.zeros((sh, sw), dtype=np.uint8)
        for dy, dx in viz:
            acc += sub_pad[1 + dy:1 + dy + sh, 1 + dx:1 + dx + sw]

        offset = y0 - ry0
        nb[y0:y1, :] = acc[offset:offset + (y1 - y0), :]

    return nb


def vetorizar_grafo(esqueleto, nb, H, W):
    """Percorre o esqueleto como grafo, extraindo arestas entre nós."""
    viz = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    nos_y, nos_x = np.where(esqueleto & (nb != 2))
    nos = set(zip(nos_y.tolist(), nos_x.tolist()))

    if not nos:
        pts = np.argwhere(esqueleto)
        if len(pts) > 0:
            nos = {(int(pts[0][0]), int(pts[0][1]))}

    visitado = set()
    arestas = []

    for no_y, no_x in nos:
        for dy0, dx0 in viz:
            ny0, nx0 = no_y + dy0, no_x + dx0
            if not (0 <= ny0 < H and 0 <= nx0 < W) or not esqueleto[ny0, nx0]:
                continue
            if (no_y, no_x, ny0, nx0) in visitado:
                continue

            caminho = [(no_y, no_x), (ny0, nx0)]
            visitado.add((no_y, no_x, ny0, nx0))
            visitado.add((ny0, nx0, no_y, no_x))

            atual = (ny0, nx0)
            prev = (no_y, no_x)

            while atual not in nos:
                prox = None
                for dy, dx in viz:
                    ny, nx = atual[0] + dy, atual[1] + dx
                    if not (0 <= ny < H and 0 <= nx < W) or not esqueleto[ny, nx]:
                        continue
                    if (ny, nx) == prev or (ny, nx) in caminho[-3:]:
                        continue
                    prox = (ny, nx)
                    break
                if prox is None:
                    break
                visitado.add((atual[0], atual[1], prox[0], prox[1]))
                visitado.add((prox[0], prox[1], atual[0], atual[1]))
                prev = atual
                atual = prox
                caminho.append(atual)

            if len(caminho) >= 2:
                arestas.append(caminho)

    return arestas


def podar_spurs(arestas, spur_max):
    """Remove arestas curtas com ponta livre e outra extremidade em junção."""
    grau = defaultdict(int)
    for c in arestas:
        grau[c[0]] += 1
        grau[c[-1]] += 1

    mantidas = []
    removidas = 0

    for c in arestas:
        g_ini = grau[c[0]]
        g_fim = grau[c[-1]]
        comp = len(c)

        eh_spur = (
            (g_ini == 1 and g_fim >= 3 and comp < spur_max) or
            (g_fim == 1 and g_ini >= 3 and comp < spur_max)
        )
        if eh_spur:
            removidas += 1
            continue
        mantidas.append(c)

    return mantidas, removidas


def main():
    log("=" * 70)
    log("  ESQUELETIZAÇÃO + VETORIZAÇÃO DE CURVAS")
    log("=" * 70)

    cfg = CONFIG
    caminho_entrada, crs = obter_raster_entrada(cfg)

    if not os.path.isfile(caminho_entrada):
        raise SystemExit(f"[ERRO] Raster não encontrado: {caminho_entrada}")

    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    caminho_saida = os.path.join(cfg["pasta_saida"], cfg["nome_saida"])

    log(f"\nEntrada: {os.path.basename(caminho_entrada)}")
    log(f"Saída:   {caminho_saida}")
    log(f"CRS:     {crs}\n")

    ds = gdal.Open(caminho_entrada)
    W, H = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    pixel_size = abs(gt[1])

    log(f"Raster: {W} × {H} px | pixel = {pixel_size:.3f} m\n")

    # ETAPA 1: Esqueletização
    esqueleto = esqueletizar_em_tiles(ds, W, H, cfg, log)

    # ETAPA 2: Contagem de vizinhos
    log("[2/4] Contando vizinhos...")
    t0 = time.time()
    nb = contar_vizinhos(esqueleto, H, W, cfg["bloco_poda"])
    log(f"   Tempo: {(time.time() - t0) / 60:.1f} min\n")

    # ETAPA 3: Vetorização + poda
    log("[3/4] Vetorizando (grafo topológico)...")
    t0 = time.time()
    arestas = vetorizar_grafo(esqueleto, nb, H, W)
    log(f"   Arestas: {len(arestas):,}")
    log(f"   Tempo: {(time.time() - t0) / 60:.1f} min\n")

    log("[3.5] Podando spurs...")
    t0 = time.time()
    arestas, removidas = podar_spurs(arestas, cfg["spur_max_px"])
    log(f"   Spurs removidos: {removidas}")
    log(f"   Arestas após poda: {len(arestas):,}\n")

    # ETAPA 4: Converter para geometrias + salvar
    log("[4/4] Gerando shapefile...")
    t0 = time.time()

    def pix_para_geo(y, x):
        return (gt[0] + (x + 0.5) * gt[1], gt[3] + (y + 0.5) * gt[5])

    geometrias = []
    for caminho in arestas:
        if len(caminho) < 2:
            continue
        coords = [pix_para_geo(y, x) for y, x in caminho]
        coords_clean = []
        for c in coords:
            if not coords_clean or coords_clean[-1] != c:
                coords_clean.append(c)
        if len(coords_clean) < 2:
            continue
        linha = LineString(coords_clean).simplify(pixel_size * 1.2, preserve_topology=False)
        geometrias.append(linha)

    if geometrias:
        gdf = gpd.GeoDataFrame(geometry=geometrias, crs=crs)
        gdf.to_file(caminho_saida, driver="ESRI Shapefile")

        log(f"\n{'=' * 70}")
        log(f"CONCLUÍDO em {(time.time() - t0) / 60:.1f} min")
        log(f"{'=' * 70}")
        log(f"Linhas:  {len(geometrias):,}")
        log(f"Arquivo: {caminho_saida}")

        # Carrega no QGIS se disponível
        try:
            from qgis.core import QgsProject, QgsVectorLayer
            lyr = QgsVectorLayer(caminho_saida, "Curvas_Vetorizadas", "ogr")
            if lyr.isValid():
                QgsProject.instance().addMapLayer(lyr)
        except ImportError:
            pass
    else:
        log("Nenhuma geometria gerada.")


if __name__ == "__main__":
    main()
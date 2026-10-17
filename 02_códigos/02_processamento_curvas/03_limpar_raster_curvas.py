# -*- coding: utf-8 -*-
"""
================================================================================
 LIMPEZA DE RASTER BINÁRIO DE CURVAS POR FILTRO DE FORMA
================================================================================

 DESCRIÇÃO
 ---------
 Remove ruído e artefatos de um raster binário de curvas (geralmente saída
 de rede neural U-Net) aplicando filtros de forma por componentes conectados.

 Cada componente conectado do raster é analisado e mantido apenas se atender
 a critérios morfológicos que caracterizam curvas de nível:
 - Área mínima (remove pontos isolados)
 - Elongação (curvas são alongadas, ruído é compacto)
 - Espessura estimada (curvas têm espessura fina e uniforme)

 FILTROS APLICADOS
 -----------------
 Para cada componente:
 - area >= AREA_MIN          → remove manchas pequenas
 - fill_ratio < FILL_MAX     → remove formas compactas (blobs)
   fill_ratio = area / (largura × altura do bbox)
 - aspect_ratio > ASPECT_MIN → exige alongamento
   aspect_ratio = max(w,h) / min(w,h)
 - espessura < THICK_MAX     → remove manchas espessas

 ESTRATÉGIA EM TILES
 -------------------
 O raster é processado em tiles com overlap para não estourar a RAM em
 rasters grandes. Componentes que cruzam bordas são capturados corretamente
 graças ao overlap.

 REQUISITOS
 ----------
 - Python 3.10+
 - GDAL, NumPy, OpenCV
 - (Opcional) QGIS Python para carregar o resultado automaticamente

 COMO USAR
 ---------
 1. Edite o dicionário CONFIG abaixo
 2. Execute: python 03_limpar_raster_curvas.py

 RESULTADO
 ---------
 Raster binário limpo (mesmo CRS e resolução do raster de entrada).

 AUTORIA
 -------
 [Seu nome] / [Programa] / [Instituição] / [Ano]
================================================================================
"""

import os
import time
import pickle

import numpy as np
import cv2
from osgeo import gdal


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

CONFIG = {
    # Caminhos
    "raster_entrada": r"CAMINHO\PARA\curvas_brutas.tif",
    "pasta_saida":    r"CAMINHO\PARA\pasta\de\saida",
    "nome_saida":     "curvas_limpas.tif",
    "nome_checkpoint": "limpeza_ckpt.pkl",

    # Filtros de forma (ajustar conforme a resolução da carta)
    "area_min":     50,     # Área mínima do componente (pixels)
    "fill_max":     0.50,   # Máximo de preenchimento do bbox (0.0 a 1.0)
    "aspect_min":   5.0,    # Mínimo de elongação (largura/altura)
    "thick_max":    999,    # Espessura máxima estimada (pixels)

    # Processamento em tiles
    "tile":         2048,   # Tamanho do tile (pixels)
    "overlap":      64,     # Sobreposição (pixels)
}


# ==============================================================================
# FUNÇÕES
# ==============================================================================

def log(msg=""):
    print(msg, flush=True)


def limpar_componente(mask, cfg):
    """
    Aplica filtro de forma por componentes conectados.

    Parâmetros:
    - mask: array numpy (H, W) binário (0 ou 255)
    - cfg: dicionário CONFIG

    Retorna: (mask_limpa, n_componentes_mantidos)
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    saida = np.zeros_like(mask)
    mantidos = 0

    for i in range(1, num):
        x, y, w, h, area = stats[i]

        if area < cfg["area_min"]:
            continue

        fill = area / float(w * h)
        aspect = max(w, h) / float(max(1, min(w, h)))
        thick = area / float(max(w, h))

        if (fill < cfg["fill_max"] or aspect > cfg["aspect_min"]) and thick < cfg["thick_max"]:
            saida[labels == i] = 255
            mantidos += 1

    return saida, mantidos


def main():
    log("=" * 70)
    log("  LIMPEZA DE RASTER DE CURVAS — FILTRO DE FORMA")
    log("=" * 70)

    cfg = CONFIG
    os.makedirs(cfg["pasta_saida"], exist_ok=True)

    caminho_saida = os.path.join(cfg["pasta_saida"], cfg["nome_saida"])
    caminho_ckpt = os.path.join(cfg["pasta_saida"], cfg["nome_checkpoint"])

    if not os.path.isfile(cfg["raster_entrada"]):
        raise SystemExit(f"[ERRO] Raster não encontrado: {cfg['raster_entrada']}")

    ds = gdal.Open(cfg["raster_entrada"])
    W, H = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    log(f"\nEntrada: {os.path.basename(cfg['raster_entrada'])}")
    log(f"Tamanho: {W} × {H} px")
    log(f"Parâmetros: AREA_MIN={cfg['area_min']}, FILL_MAX={cfg['fill_max']}, "
        f"ASPECT_MIN={cfg['aspect_min']}\n")

    # Cria saída
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        caminho_saida, W, H, 1, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    # Checkpoint
    tile_ini = 0
    if os.path.exists(caminho_ckpt):
        with open(caminho_ckpt, "rb") as f:
            tile_ini = pickle.load(f)["tile"]
        log(f"[RETOMANDO] do tile {tile_ini}")

    # Lista de tiles
    step = cfg["tile"] - cfg["overlap"]
    tiles = [(x, y) for y in range(0, H, step) for x in range(0, W, step)]
    log(f"Tiles: {len(tiles)}\n")

    t0 = time.time()
    tot_antes = 0
    tot_depois = 0
    tot_mant = 0

    for k in range(tile_ini, len(tiles)):
        x, y = tiles[k]

        rx = max(0, x - cfg["overlap"])
        ry = max(0, y - cfg["overlap"])
        rw = min(cfg["tile"] + 2 * cfg["overlap"], W - rx)
        rh = min(cfg["tile"] + 2 * cfg["overlap"], H - ry)

        arr = ds.GetRasterBand(1).ReadAsArray(rx, ry, rw, rh)
        if arr is None:
            continue

        mask = (arr > 0).astype(np.uint8) * 255
        tot_antes += int((mask > 0).sum())

        limpo, mant = limpar_componente(mask, cfg)
        tot_mant += mant

        cx1, cy1 = x - rx, y - ry
        cw = min(cfg["tile"], W - x)
        ch = min(cfg["tile"], H - y)
        core = limpo[cy1:cy1 + ch, cx1:cx1 + cw]

        tot_depois += int((core > 0).sum())
        band.WriteArray((core > 0).astype(np.uint8), x, y)

        if (k + 1) % 20 == 0 or k == len(tiles) - 1:
            dt = (time.time() - t0) / 60
            feitos = k + 1 - tile_ini
            resta = dt / max(1, feitos) * (len(tiles) - tile_ini - feitos)
            log(f"  {k+1}/{len(tiles)} | {dt:.1f} min | resta ~{resta:.1f} min")

        if (k + 1) % 100 == 0:
            with open(caminho_ckpt, "wb") as f:
                pickle.dump({"tile": k + 1}, f)

    band.FlushCache()
    out_ds = None
    ds = None

    if os.path.exists(caminho_ckpt):
        os.remove(caminho_ckpt)

    log(f"\n{'=' * 70}")
    log(f"CONCLUÍDO em {(time.time() - t0) / 60:.1f} min")
    log(f"{'=' * 70}")
    log(f"Pixels antes:   {tot_antes:,}")
    log(f"Pixels depois:  {tot_depois:,}")
    log(f"Redução:        {100 * (1 - tot_depois / max(1, tot_antes)):.1f}%")
    log(f"Componentes:    {tot_mant:,}")
    log(f"Arquivo:        {caminho_saida}")

    # Tenta carregar no QGIS (se disponível)
    try:
        from qgis.core import QgsProject, QgsRasterLayer
        lyr = QgsRasterLayer(caminho_saida, "Curvas_Limpas", "gdal")
        if lyr.isValid():
            QgsProject.instance().addMapLayer(lyr)
            log("Camada carregada no QGIS")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
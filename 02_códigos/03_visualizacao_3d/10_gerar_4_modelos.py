# -*- coding: utf-8 -*-
"""
================================================================================
 GERAÇÃO DE 4 MODELOS OBJ — MÁXIMA DEFINIÇÃO + README
================================================================================

 Gera 4 modelos tridimensionais em OBJ a partir de um MDT.
 Os modelos são CENTRADOS na origem (0,0,0) para uso direto em Rhino.

 Modelos gerados:
   1. geometria pura       — escala real, sem textura
   2. texturizado          — escala real, com textura
   3. exagerado            — Z × 3, sem textura
   4. exagerado texturizado — Z × 3, com textura

 PARÂMETROS DE DEFINIÇÃO
 -----------------------
 - passo = 1  →  MÁXIMA definição (2,2 M vértices, arquivo ~170 MB)
 - passo = 2  →  BOA definição (560 mil vértices, arquivo ~44 MB)
 - passo = 3  →  leve (250 mil vértices, arquivo ~20 MB)

 Sem suavização → mantém 100% da volumetria do MDT
================================================================================
"""

import os
import numpy as np
from osgeo import gdal
from PIL import Image


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

CONFIG = {
    # Caminhos
    "mdt":         r"D:\Documents\Mestrado FAU Matheus\04. AULAS\Arquiologia da Paisagem\MDT Final\Produtos MDT\SARA BRASIL 1930 MDT 10m.tif",
    "textura":     r"D:\Documents\Mestrado FAU Matheus\04. AULAS\Arquiologia da Paisagem\qgis\testura.tif",
    "pasta_saida": r"D:\Documents\Mestrado FAU Matheus\04. AULAS\Arquiologia da Paisagem\00_PROJETO RECONSTRUÇÃO DO RELEVO PAULISTANO\07_modelos_3d",

    # DEFINIÇÃO — escolha aqui
    "passo":             1,      # 1 = máx | 2 = boa | 3 = leve

    # Suavização (0 = fiel ao dado; >0 = mais suave)
    "suavizar_sigma":    0.0,

    # Exagero vertical (modelos 3 e 4)
    "z_exagero":         1.5,

    # Textura
    "textura_tam":       4096,
}


# ==============================================================================
# FUNÇÕES
# ==============================================================================

def log(msg=""):
    print(msg, flush=True)


def ler_mdt(caminho):
    ds = gdal.Open(caminho)
    if ds is None:
        raise SystemExit(f"[ERRO] Não abri: {caminho}")
    gt = ds.GetGeoTransform()
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    valido = np.isfinite(arr) & (arr > -1000)
    if nodata is not None:
        valido &= (arr != nodata)
    return arr, valido, gt


def preparar_textura(textura_path, mdt_path, pasta_saida, tamanho):
    log("   Alinhando textura ao MDT...")
    ds_mdt = gdal.Open(mdt_path)
    gt = ds_mdt.GetGeoTransform()
    w, h = ds_mdt.RasterXSize, ds_mdt.RasterYSize
    proj = ds_mdt.GetProjection()
    ds_mdt = None

    tmp = os.path.join(pasta_saida, "_tmp_textura.tif")
    gdal.Warp(
        tmp, textura_path, format="GTiff",
        width=w, height=h,
        outputBounds=(gt[0], gt[3] + h * gt[5], gt[0] + w * gt[1], gt[3]),
        dstSRS=proj, resampleAlg="bilinear",
        creationOptions=["COMPRESS=LZW"],
    )
    ds = gdal.Open(tmp)
    n = ds.RasterCount
    if n >= 3:
        r = ds.GetRasterBand(1).ReadAsArray()
        g = ds.GetRasterBand(2).ReadAsArray()
        b = ds.GetRasterBand(3).ReadAsArray()
        rgb = np.dstack([r, g, b]).astype(np.uint8)
    else:
        arr = ds.GetRasterBand(1).ReadAsArray()
        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
        rgb = np.dstack([arr, arr, arr])
    ds = None

    img = Image.fromarray(rgb).resize((tamanho, tamanho), Image.LANCZOS)
    png = os.path.join(pasta_saida, "textura.png")
    img.save(png)
    if os.path.exists(tmp):
        os.remove(tmp)
    log(f"   textura.png ({tamanho}×{tamanho})")


def gerar_obj(caminho_obj, arr_d, mascara, res, z_exagero,
              usar_textura, nome_png=None):
    Hd, Wd = arr_d.shape
    z_min = arr_d[mascara].min()
    z_max = arr_d[mascara].max()
    z_norm = np.zeros_like(arr_d)
    z_norm[mascara] = (arr_d[mascara] - z_min) / (z_max - z_min) * z_exagero * 100

    ix_c = (Wd - 1) / 2.0
    iy_c = (Hd - 1) / 2.0

    verts = []
    uvs = []
    idx = {}
    for iy in range(Hd):
        for ix in range(Wd):
            if not mascara[iy, ix]:
                continue
            x = (ix - ix_c) * res
            y = (iy_c - iy) * res
            z = float(z_norm[iy, ix])
            idx[(iy, ix)] = len(verts) + 1
            verts.append((x, y, z))
            uvs.append((ix / (Wd - 1), 1 - iy / (Hd - 1)))

    faces = []
    for iy in range(Hd - 1):
        for ix in range(Wd - 1):
            c = [(iy, ix), (iy, ix + 1), (iy + 1, ix + 1), (iy + 1, ix)]
            if not all(k in idx for k in c):
                continue
            v0, v1, v2, v3 = [idx[k] for k in c]
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))

    with open(caminho_obj, "w") as f:
        nb = os.path.basename(caminho_obj)
        f.write(f"# MDT Sara Brasil 1930 — {nb}\n")
        f.write(f"# Centrado em (0,0,0) | pixel {res:.2f} m\n")
        f.write(f"# Exagero vertical: {z_exagero}\n")
        f.write(f"# Vértices: {len(verts):,} | Faces: {len(faces):,}\n")
        if usar_textura and nome_png:
            f.write(f"mtllib {nb.replace('.obj', '.mtl')}\n")
        f.write("\n")
        for v in verts:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.4f}\n")
        if usar_textura:
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            f.write(f"\nusemtl maquete\n")
        for v0, v1, v2 in faces:
            if usar_textura:
                f.write(f"f {v0}/{v0} {v1}/{v1} {v2}/{v2}\n")
            else:
                f.write(f"f {v0} {v1} {v2}\n")

    if usar_textura and nome_png:
        with open(caminho_obj.replace(".obj", ".mtl"), "w") as f:
            f.write(f"newmtl maquete\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\n")
            f.write(f"Ks 0.0 0.0 0.0\nd 1.0\nmap_Kd {nome_png}\n")

    tam = os.path.getsize(caminho_obj) / 1e6
    return len(verts), len(faces), tam




# ==============================================================================
# EXECUÇÃO
# ==============================================================================

def main():
    cfg = CONFIG
    os.makedirs(cfg["pasta_saida"], exist_ok=True)

    log("=" * 70)
    log("  GERAÇÃO DE 4 MODELOS OBJ")
    log("=" * 70)

    log("\n[1/4] Lendo MDT...")
    arr, valido, gt = ler_mdt(cfg["mdt"])
    H, W = arr.shape
    log(f"   {W} × {H} px | pixel {gt[1]:.2f} m")

    if cfg["suavizar_sigma"] > 0:
        from scipy.ndimage import gaussian_filter
        log(f"   Suavizando (σ = {cfg['suavizar_sigma']})")
        arr = gaussian_filter(arr, sigma=cfg["suavizar_sigma"])
    else:
        log("   Sem suavização (fiel ao dado)")

    if cfg["passo"] > 1:
        log(f"\n[2/4] Reduzindo (passo = {cfg['passo']})...")
        arr = arr[::cfg["passo"], ::cfg["passo"]]
        valido = valido[::cfg["passo"], ::cfg["passo"]]
        res = gt[1] * cfg["passo"]
        H, W = arr.shape
        log(f"   {W} × {H} px | pixel {res:.2f} m")
    else:
        res = gt[1]
        log(f"\n[2/4] Sem downsample ({W} × {H} px | pixel {res:.2f} m)")

    arr = np.where(valido, arr, 0)

    log("\n[3/4] Gerando 4 modelos...")
    pasta = cfg["pasta_saida"]
    z = cfg["z_exagero"]

    log("\n[A] Modelo 1 — Geometria pura")
    p = os.path.join(pasta, "modelo_1_geometria.obj")
    nv, nf, tam = gerar_obj(p, arr, valido, res, 1.0, False)
    log(f"   {nv:,} v | {nf:,} f | {tam:.1f} MB")

    log("\n[B] Preparando textura...")
    preparar_textura(cfg["textura"], cfg["mdt"], pasta, cfg["textura_tam"])

    log("\n[C] Modelo 2 — Texturizado")
    p = os.path.join(pasta, "modelo_2_texturizado.obj")
    nv, nf, tam = gerar_obj(p, arr, valido, res, 1.0, True, "textura.png")
    log(f"   {nv:,} v | {nf:,} f | {tam:.1f} MB")

    log(f"\n[D] Modelo 3 — Exagerado (Z × {z})")
    p = os.path.join(pasta, "modelo_3_exagerado.obj")
    nv, nf, tam = gerar_obj(p, arr, valido, res, z, False)
    log(f"   {nv:,} v | {nf:,} f | {tam:.1f} MB")

    log(f"\n[E] Modelo 4 — Exagerado + texturizado")
    p = os.path.join(pasta, "modelo_4_exagerado_texturizado.obj")
    nv, nf, tam = gerar_obj(p, arr, valido, res, z, True, "textura.png")
    log(f"   {nv:,} v | {nf:,} f | {tam:.1f} MB")


    log(f"\n{'=' * 70}")
    log("CONCLUÍDO")
    log(f"{'=' * 70}")
    log(f"Pasta: {pasta}")


if __name__ == "__main__":
    main()
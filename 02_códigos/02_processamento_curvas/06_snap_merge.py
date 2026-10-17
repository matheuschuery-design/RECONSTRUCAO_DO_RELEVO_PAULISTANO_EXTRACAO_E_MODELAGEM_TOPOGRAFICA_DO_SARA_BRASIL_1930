# -*- coding: utf-8 -*-
"""
================================================================================
 SNAP DE EXTREMIDADES E MERGE DE LINHAS EM TILES
================================================================================

 DESCRIÇÃO
 ---------
 Consolida uma camada de linhas vetoriais aplicando duas operações:

 1. SNAP DE EXTREMIDADES
    Pontas de linhas que estão a menos de SNAP_TOL_M entre si são "puxadas"
    para um ponto médio comum, garantindo conectividade topológica.

 2. MERGE (linemerge)
    Segmentos que compartilham extremidades são unidos em linhas contínuas
    únicas, reduzindo a fragmentação.

 ESTRATÉGIA EM TILES
 -------------------
 Para lidar com camadas grandes (centenas de milhares de feições), o script
 processa a área em tiles com overlap. Em cada tile:
 - As linhas dentro do tile (e do overlap) são carregadas
 - O snap e o merge são aplicados localmente
 - Apenas o resultado no núcleo do tile é mantido
 - Ao final, uma consolidação global une tudo

 DETECÇÃO DE PONTAS PARA SNAP
 ----------------------------
 Para cada linha, suas duas extremidades são coletadas. Um KDTree identifica
 pares de extremidades dentro de SNAP_TOL_M. Cada grupo de extremidades
 próximas é unificado em um ponto médio (via Union-Find).

 APLICAÇÃO
 ---------
 Indicado após a vetorização + conexão de gaps, quando ainda há muitas
 linhas fragmentadas com extremidades ligeiramente desalinhadas.

 REQUISITOS
 ----------
 - Python 3.10+
 - GeoPandas, Shapely, SciPy, NumPy
 - pyogrio (opcional, para leitura rápida de bbox)

 COMO USAR
 ---------
 1. Edite o dicionário CONFIG abaixo
 2. Execute: python 06_snap_merge.py

 RESULTADO
 ---------
 GeoPackage (.gpkg) com linhas consolidadas (menos feições, mais contínuas).

 AUTORIA
 -------
 [Seu nome] / [Programa] / [Instituição] / [Ano]
================================================================================
"""

import os
import sys
import time
import gc
from collections import defaultdict

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import unary_union, linemerge
from scipy.spatial import cKDTree


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

CONFIG = {
    # Caminhos
    "gpkg_entrada": r"CAMINHO\PARA\linhas_conectadas.gpkg",
    "pasta_saida":  r"CAMINHO\PARA\pasta\de\saida",
    "sufixo_saida": "_merged",

    # Tiles
    "tile_m":          1500.0,   # Tamanho do tile (metros)
    "overlap_m":       60.0,     # Sobreposição (maior que snap)
    "snap_tol_m":      1.5,      # Tolerância do snap (metros)
    "min_len_save":    0.05,     # Comprimento mínimo para salvar (metros)
    "explode":         True,     # Se True, explode MultiLineString
}


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def log(msg=""):
    print(msg, flush=True)


def obter_bbox(caminho):
    """Obtém bbox e número de feições do arquivo de entrada."""
    # Tenta pyogrio (rápido)
    try:
        import pyogrio
        info = pyogrio.read_info(caminho)
        return info["total_bounds"], info["features"]
    except (ImportError, Exception):
        pass

    # Fallback: leitura em chunks
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    n = 0
    for chunk in gpd.read_file(caminho, chunksize=100000):
        b = chunk.total_bounds
        minx = min(minx, b[0])
        miny = min(miny, b[1])
        maxx = max(maxx, b[2])
        maxy = max(maxy, b[3])
        n += len(chunk)
    return (minx, miny, maxx, maxy), n


def aplicar_snap_e_merge(linhas, snap_tol):
    """
    Aplica snap de extremidades e merge (linemerge) em um conjunto de linhas.

    Parâmetros:
    - linhas: lista de LineStrings
    - snap_tol: tolerância de snap (mesmas unidades das coordenadas)

    Retorna: lista de LineStrings consolidados.
    """
    if len(linhas) < 2:
        return list(linhas)

    # Coleta todas as extremidades
    pontas = []
    for i, l in enumerate(linhas):
        coords = list(l.coords)
        if len(coords) < 2:
            continue
        pontas.append((i, "start", np.array(coords[0][:2])))
        pontas.append((i, "end",   np.array(coords[-1][:2])))

    if len(pontas) < 2:
        return list(linhas)

    # KDTree + Union-Find para agrupar pontas próximas
    coords_arr = np.array([p[2] for p in pontas])
    tree = cKDTree(coords_arr)
    pares = tree.query_pairs(snap_tol, output_type="ndarray")

    pai = list(range(len(pontas)))

    def find(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            pai[ra] = rb

    for i, j in pares:
        union(int(i), int(j))

    # Agrupa pontas por raiz
    grupos = defaultdict(list)
    for k in range(len(pontas)):
        grupos[find(k)].append(k)

    # Ponto médio para cada grupo
    nova_pos = {}
    for g, membros in grupos.items():
        if len(membros) < 2:
            continue
        pts = np.array([pontas[k][2] for k in membros])
        centro = pts.mean(axis=0)
        for k in membros:
            nova_pos[k] = centro

    # Aplica snap
    snap_dict = {}
    for k, (idx_l, end, _) in enumerate(pontas):
        if k in nova_pos:
            snap_dict[(idx_l, end)] = tuple(nova_pos[k].tolist())

    linhas_snap = []
    for i, l in enumerate(linhas):
        coords = list(l.coords)
        if len(coords) < 2:
            linhas_snap.append(l)
            continue

        if (i, "start") in snap_dict:
            novo = snap_dict[(i, "start")]
            coords[0] = novo + tuple(coords[0][2:])
        if (i, "end") in snap_dict:
            novo = snap_dict[(i, "end")]
            coords[-1] = novo + tuple(coords[-1][2:])

        linhas_snap.append(LineString(coords))

    # Linemerge
    try:
        u = unary_union(linhas_snap)
        if u.geom_type == "LineString":
            merged = u
        elif u.geom_type == "MultiLineString":
            merged = linemerge(u)
        else:
            parts = [g for g in u.geoms
                     if g.geom_type in ("LineString", "MultiLineString")]
            merged = linemerge(unary_union(parts)) if parts else u
    except Exception:
        return linhas_snap

    # Explode
    if merged.geom_type == "LineString":
        return [merged]
    elif merged.geom_type == "MultiLineString":
        return [g for g in merged.geoms if not g.is_empty]
    elif hasattr(merged, "geoms"):
        return [g for g in merged.geoms
                if g.geom_type == "LineString" and not g.is_empty]
    return [merged]


def main():
    log("=" * 70)
    log("  SNAP + MERGE DE LINHAS")
    log("=" * 70)

    cfg = CONFIG
    os.makedirs(cfg["pasta_saida"], exist_ok=True)

    base = os.path.splitext(os.path.basename(cfg["gpkg_entrada"]))[0]
    caminho_saida = os.path.join(
        cfg["pasta_saida"], f"{base}{cfg['sufixo_saida']}.gpkg"
    )

    if not os.path.isfile(cfg["gpkg_entrada"]):
        raise SystemExit(f"[ERRO] Entrada não encontrada: {cfg['gpkg_entrada']}")

    log(f"\nEntrada: {os.path.basename(cfg['gpkg_entrada'])}")
    log(f"Saída:   {os.path.basename(caminho_saida)}")
    log(f"Tile: {cfg['tile_m']} m | Snap: {cfg['snap_tol_m']} m\n")

    # [1/5] BBox
    log("[1/5] Descobrindo bbox...")
    t0 = time.time()
    (minx, miny, maxx, maxy), n_total = obter_bbox(cfg["gpkg_entrada"])
    log(f"   Total: {n_total:,} feições")
    log(f"   BBox: {maxx - minx:.0f} × {maxy - miny:.0f} m")
    log(f"   Tempo: {time.time() - t0:.1f}s\n")

    # [2/5] Grade
    nx = int(np.ceil((maxx - minx) / cfg["tile_m"]))
    ny = int(np.ceil((maxy - miny) / cfg["tile_m"]))
    n_tiles = nx * ny
    log(f"[2/5] Grade: {nx} × {ny} = {n_tiles} tiles\n")

    # [3/5] Processa cada tile
    log("[3/5] Processando tiles...")
    t0 = time.time()

    resultados = []
    n_vazios = 0
    ov = cfg["overlap_m"]

    for iy in range(ny):
        for ix in range(nx):
            tx0 = minx + ix * cfg["tile_m"]
            ty0 = miny + iy * cfg["tile_m"]
            tx1 = tx0 + cfg["tile_m"]
            ty1 = ty0 + cfg["tile_m"]

            # BBox com overlap
            qx0 = tx0 - ov
            qy0 = ty0 - ov
            qx1 = tx1 + ov
            qy1 = ty1 + ov

            try:
                chunk = gpd.read_file(cfg["gpkg_entrada"], bbox=(qx0, qy0, qx1, qy1))
            except Exception as e:
                log(f"   tile {ix},{iy}: {e}")
                continue

            if len(chunk) == 0:
                n_vazios += 1
                continue

            # Extrai LineStrings
            linhas = []
            for g in chunk.geometry:
                if g is None or g.is_empty:
                    continue
                if g.geom_type == "LineString":
                    linhas.append(g)
                elif g.geom_type == "MultiLineString":
                    linhas.extend([s for s in g.geoms if not s.is_empty])

            if not linhas:
                n_vazios += 1
                continue

            # Aplica snap + merge
            merged = aplicar_snap_e_merge(linhas, cfg["snap_tol_m"])

            # Recorta na caixa do tile (sem overlap)
            try:
                tile_box = box(tx0, ty0, tx1, ty1)
                merged = [g.intersection(tile_box) for g in merged]
                merged = [g for g in merged if g and not g.is_empty]
            except Exception:
                pass

            # Adiciona LineStrings
            for g in merged:
                if g.geom_type == "LineString" and g.length >= cfg["min_len_save"]:
                    resultados.append(g)
                elif g.geom_type == "MultiLineString":
                    for sub in g.geoms:
                        if sub.length >= cfg["min_len_save"]:
                            resultados.append(sub)

            del chunk, linhas, merged
            gc.collect()

            k = iy * nx + ix + 1
            if k % 5 == 0 or k == n_tiles:
                dt = time.time() - t0
                resta = dt / max(1, k) * (n_tiles - k)
                log(f"   tile {k}/{n_tiles} | linhas: {len(resultados):,} | resta ~{resta:.0f}s")

    log(f"\n   Tiles vazios: {n_vazios}")
    log(f"   Linhas acumuladas: {len(resultados):,}")
    log(f"   Tempo: {time.time() - t0:.1f}s\n")

    # [4/5] Consolidação global
    log("[4/5] Consolidando...")
    t0 = time.time()

    if not resultados:
        raise SystemExit("[ERRO] Nada gerado.")

    try:
        u = unary_union(resultados)
        if u.geom_type == "LineString":
            merged = u
        elif u.geom_type == "MultiLineString":
            merged = linemerge(u)
        else:
            parts = [g for g in u.geoms
                     if g.geom_type in ("LineString", "MultiLineString")]
            merged = linemerge(unary_union(parts)) if parts else u
    except Exception as e:
        log(f"   unary_union falhou ({e})")
        merged = MultiLineString(resultados) if len(resultados) > 1 else resultados[0]

    log(f"   Tipo após merge: {merged.geom_type}  ({time.time() - t0:.1f}s)")

    # [5/5] Explode e salva
    log("\n[5/5] Salvando...")

    if cfg["explode"]:
        if merged.geom_type == "LineString":
            finais = [merged]
        elif merged.geom_type == "MultiLineString":
            finais = [g for g in merged.geoms if not g.is_empty]
        else:
            finais = [g for g in getattr(merged, "geoms", [])
                      if g.geom_type == "LineString" and not g.is_empty]
    else:
        finais = [merged]

    finais = [g for g in finais if g.length >= cfg["min_len_save"]]

    if not finais:
        raise SystemExit("[ERRO] Nada sobrou após filtragem.")

    # Remove arquivo existente
    if os.path.exists(caminho_saida):
        try:
            os.remove(caminho_saida)
        except Exception:
            pass
    for ext in ("-journal", "-wal"):
        p = caminho_saida + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    # Obtém CRS do original
    head = gpd.read_file(cfg["gpkg_entrada"], rows=1)
    crs = head.crs.to_string() if head.crs else "EPSG:31983"

    gdf_out = gpd.GeoDataFrame(geometry=finais, crs=crs)
    gdf_out.to_file(caminho_saida, driver="GPKG")

    comps = np.array([g.length for g in finais])
    log(f"\n{'=' * 70}")
    log(f"CONCLUÍDO em {(time.time() - t0) / 60:.1f} min")
    log(f"{'=' * 70}")
    log(f"Entrada:  {n_total:,} feições")
    log(f"Saída:    {len(finais):,} linhas")
    log(f"Redução:  {100 * (1 - len(finais) / max(1, n_total)):.1f}%")
    log(f"Comprimentos (m):")
    log(f"   min:   {comps.min():.2f}")
    log(f"   med:   {np.percentile(comps, 50):.2f}")
    log(f"   max:   {comps.max():.2f}")
    log(f"   <5m:   {int((comps < 5).sum()):,}")
    log(f"   >50m:  {int((comps > 50).sum()):,}")
    log(f"   >100m: {int((comps > 100).sum()):,}")
    log(f"\nArquivo: {caminho_saida}")


if __name__ == "__main__":
    main()
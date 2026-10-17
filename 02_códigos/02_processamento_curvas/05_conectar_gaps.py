# -*- coding: utf-8 -*-
"""
================================================================================
 CONEXÃO DE GAPS ENTRE PONTAS DE CURVAS DE NÍVEL
================================================================================

 DESCRIÇÃO
 ---------
 Conecta automaticamente pontas soltas de curvas de nível que deveriam estar
 unidas, mas foram separadas durante o processo de vetorização (por falhas na
 carta original, ruído na esqueletização ou sobreposição).

 CRITÉRIO DE CONEXÃO
 -------------------
 Duas pontas são conectadas se:
 1. A distância entre elas for menor que RAIO_MAX_M
 2. Os vetores de saída das duas curvas apontarem uma para a outra,
    com ângulo menor que ANGULO_MAX_DEG

 O vetor de saída de cada ponta é estimado pela direção dos N primeiros
 pontos da linha a partir da extremidade (parâmetro "n_pontos_vetor").

 USO DE ÁRVORE ESPACIAL (KDTree)
 --------------------------------
 A busca de pontas próximas usa KDTree para eficiência (O(n log n) em vez
 de O(n²)). Isso permite processar milhares de pontas em segundos.

 APLICAÇÃO
 ---------
 Útil quando o raster de curvas tem falhas nas interseções, ou quando a
 vetorização separa segmentos que deveriam ser contínuos.

 REQUISITOS
 ----------
 - Python 3.10+
 - GeoPandas, Shapely, SciPy, NumPy
 - QGIS Python (opcional)

 COMO USAR
 ---------
 1. Edite o dicionário CONFIG abaixo
 2. Execute: python 05_conectar_gaps.py

 RESULTADO
 ---------
 Novo shapefile com as pontas conectadas + merge final (linhas contínuas).

 AUTORIA
 -------
 [Seu nome] / [Programa] / [Instituição] / [Ano]
================================================================================
"""

import os
import time

import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union, linemerge


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

CONFIG = {
    # Caminhos
    "shp_entrada": r"CAMINHO\PARA\linhas_brutas.shp",
    "pasta_saida": r"CAMINHO\PARA\pasta\de\saida",
    "nome_saida":  "linhas_conectadas.shp",

    # Parâmetros de conexão
    "raio_max_m":     15.0,   # Distância máxima entre pontas (metros)
    "angulo_max_deg": 25.0,   # Ângulo máximo entre vetor de saída e a ponte
    "n_pontos_vetor": 5,      # Quantos pontos usar para estimar o vetor
    "tolerancia_enc": 0.4,    # Tolerância para considerar duas pontas "já unidas" (m)
}


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def log(msg=""):
    print(msg, flush=True)


def extrair_linestrings(gdf):
    """Extrai todas as LineStrings de um GeoDataFrame."""
    linhas = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            linhas.append(geom)
        elif geom.geom_type == "MultiLineString":
            linhas.extend([s for s in geom.geoms if not s.is_empty])
    return linhas


def identificar_pontas(linhas, tolerancia, n_pontos):
    """
    Identifica pontas livres (extremidades não compartilhadas por outra linha).

    Retorna lista de dicionários com:
    - coord: (x, y) da ponta
    - vetor: direção unitária apontando para fora da linha
    - aresta_id: índice da linha de origem
    """
    # Coleta todas as extremidades
    extremidades = []
    for idx, l in enumerate(linhas):
        coords = list(l.coords)
        if len(coords) < 2:
            continue
        extremidades.append({
            "aresta_id": idx,
            "tipo": "start",
            "coord": np.array(coords[0][:2]),
            "coords": coords,
        })
        extremidades.append({
            "aresta_id": idx,
            "tipo": "end",
            "coord": np.array(coords[-1][:2]),
            "coords": coords,
        })

    if not extremidades:
        return []

    # Encontra extremidades compartilhadas (mesmo ponto em linhas diferentes)
    coords_arr = np.array([e["coord"] for e in extremidades])
    tree = cKDTree(coords_arr)

    pontas = []
    for i, e in enumerate(extremidades):
        vizinhos = tree.query_ball_point(e["coord"], tolerancia)
        outros = [j for j in vizinhos
                  if j != i and extremidades[j]["aresta_id"] != e["aresta_id"]]

        if outros:
            continue  # já compartilhada

        # Calcula vetor de saída
        coords = e["coords"]
        if len(coords) < n_pontos + 1:
            continue

        if e["tipo"] == "start":
            c0 = np.array(coords[0][:2])
            cN = np.array(coords[n_pontos][:2])
        else:
            c0 = np.array(coords[-1][:2])
            cN = np.array(coords[-n_pontos - 1][:2])

        v = c0 - cN
        n = np.linalg.norm(v)
        if n == 0:
            continue

        pontas.append({
            "coord": tuple(c0),
            "vetor": v / n,
            "aresta_id": e["aresta_id"],
        })

    return pontas


def conectar(pontas, raio_max, angulo_max):
    """
    Emparelha pontas próximas com base em distância e ângulo.

    Retorna lista de pontes (pares de coordenadas).
    """
    if not pontas:
        return []

    coords_arr = np.array([p["coord"] for p in pontas])
    tree = cKDTree(coords_arr)

    usados = set()
    pontes = []

    for i, p1 in enumerate(pontas):
        if i in usados:
            continue

        vizinhos = tree.query_ball_point(coords_arr[i], raio_max)
        melhor = None
        melhor_custo = float("inf")

        for j in vizinhos:
            if i == j or j in usados:
                continue
            if p1["aresta_id"] == pontas[j]["aresta_id"]:
                continue

            p2 = pontas[j]
            dy = p2["coord"][0] - p1["coord"][0]
            dx = p2["coord"][1] - p1["coord"][1]
            dist = (dy * dy + dx * dx) ** 0.5
            if dist == 0:
                continue

            v_dir = np.array([dy / dist, dx / dist])

            dot1 = np.clip(np.dot(p1["vetor"], v_dir), -1, 1)
            dot2 = np.clip(np.dot(p2["vetor"], -v_dir), -1, 1)

            a1 = np.degrees(np.arccos(dot1))
            a2 = np.degrees(np.arccos(dot2))

            if a1 > angulo_max or a2 > angulo_max:
                continue

            custo = dist + (a1 + a2) * 2
            if custo < melhor_custo:
                melhor_custo = custo
                melhor = j

        if melhor is not None:
            p2 = pontas[melhor]
            pontes.append([p1["coord"], p2["coord"]])
            usados.add(i)
            usados.add(melhor)

    return pontes


def main():
    log("=" * 70)
    log("  CONEXÃO DE GAPS ENTRE CURVAS")
    log("=" * 70)

    cfg = CONFIG
    os.makedirs(cfg["pasta_saida"], exist_ok=True)
    caminho_saida = os.path.join(cfg["pasta_saida"], cfg["nome_saida"])

    if not os.path.isfile(cfg["shp_entrada"]):
        raise SystemExit(f"[ERRO] Entrada não encontrada: {cfg['shp_entrada']}")

    log(f"\nEntrada: {os.path.basename(cfg['shp_entrada'])}")
    log(f"Raio máximo: {cfg['raio_max_m']} m")
    log(f"Ângulo máximo: {cfg['angulo_max_deg']}°\n")

    # [1/3] Carrega linhas
    log("[1/3] Lendo linhas...")
    gdf = gpd.read_file(cfg["shp_entrada"])
    crs = gdf.crs.to_string() if gdf.crs else "EPSG:31983"
    linhas = extrair_linestrings(gdf)
    log(f"   {len(linhas):,} LineStrings\n")

    # [2/3] Identifica pontas
    log("[2/3] Identificando pontas livres...")
    t0 = time.time()
    pontas = identificar_pontas(linhas, cfg["tolerancia_enc"], cfg["n_pontos_vetor"])
    log(f"   {len(pontas):,} pontas  ({(time.time() - t0):.1f}s)\n")

    # [3/3] Conecta
    log("[3/3] Conectando pontas...")
    t0 = time.time()
    pontes = conectar(pontas, cfg["raio_max_m"], cfg["angulo_max_deg"])
    log(f"   {len(pontes)} pontes  ({(time.time() - t0):.1f}s)\n")

    # Monta geometrias finais
    log("Mesclando...")
    geoms_final = list(linhas)
    for (p1, p2) in pontes:
        geoms_final.append(LineString([p1, p2]))

    log(f"   Antes do merge: {len(geoms_final):,}")

    if geoms_final:
        u = unary_union(geoms_final)
        try:
            m = linemerge(u)
            if m.geom_type == "LineString":
                geoms_final = [m]
            elif m.geom_type == "MultiLineString":
                geoms_final = list(m.geoms)
        except Exception:
            pass

    log(f"   Após merge: {len(geoms_final):,}")

    # Salva
    gdf_out = gpd.GeoDataFrame(geometry=geoms_final, crs=crs)
    gdf_out.to_file(caminho_saida, driver="ESRI Shapefile")

    log(f"\n{'=' * 70}")
    log(f"CONCLUÍDO")
    log(f"Arquivo: {caminho_saida}")

    # Carrega no QGIS se disponível
    try:
        from qgis.core import QgsProject, QgsVectorLayer
        lyr = QgsVectorLayer(caminho_saida, "Linhas_Conectadas", "ogr")
        if lyr.isValid():
            QgsProject.instance().addMapLayer(lyr)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
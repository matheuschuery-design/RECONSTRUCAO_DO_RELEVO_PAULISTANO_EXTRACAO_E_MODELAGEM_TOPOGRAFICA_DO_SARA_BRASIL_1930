# -*- coding: utf-8 -*-
"""
================================================================================
 REMOÇÃO DE GALHOS (ESPORÕES) EM LINHAS VETORIAIS
================================================================================

 DESCRIÇÃO
 ---------
 Remove pequenos "galhos" (esporões / spurs) de uma camada de linhas
 vetoriais — segmentos curtos pendurados em junções, geralmente resultantes
 de imperfeições na esqueletização ou de fragmentos isolados da carta.

 TIPOS DE GALHO REMOVIDOS
 ------------------------
 O script identifica e remove dois tipos:

 1. GALHO PENDURADO
    Segmento com:
    - Uma extremidade livre (grau 1)
    - Outra extremidade em uma junção (grau ≥ 3)
    - Comprimento < GALHO_MAX_M

 2. SEGMENTO ISOLADO
    Segmento com:
    - Ambas extremidades livres (grau 1)
    - Comprimento < GALHO_MAX_M

 CÁLCULO DO GRAU
 ---------------
 O grau de cada extremidade é o número de vezes que ela aparece em outras
 linhas da camada. Duas extremidades que estão a menos de TOLERÂNCIA_ENC
 metros são consideradas o mesmo ponto.

 APLICAÇÃO
 ---------
 Útil como última etapa de limpeza, após snap+merge, para remover os
 fragmentos remanescentes que poluiriam a análise visual.

 REQUISITOS
 ----------
 - Python 3.10+
 - GeoPandas, Shapely, NumPy
 - QGIS Python (opcional)

 COMO USAR
 ---------
 1. Edite o dicionário CONFIG abaixo
 2. Execute: python 07_remover_galhos.py

 RESULTADO
 ---------
 GeoPackage (.gpkg) com galhos removidos.

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


# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

CONFIG = {
    # Caminhos
    "gpkg_entrada": r"CAMINHO\PARA\linhas_merged.gpkg",
    "pasta_saida":  r"CAMINHO\PARA\pasta\de\saida",
    "sufixo_saida": "_sem_galhos",

    # Parâmetros
    "galho_max_m":    3.0,      # Comprimento máximo de galho (metros)
    "tolerancia_enc": 0.1,      # Tolerância para coincidência de pontos (m)
    "carregar_qgis":  True,     # Carregar resultado no QGIS
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
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            linhas.append(geom)
        elif geom.geom_type == "MultiLineString":
            linhas.extend([s for s in geom.geoms if not s.is_empty])
    return linhas


def chave_ponto(pt, tol):
    """Cria uma chave quantizada de um ponto (para agrupamento)."""
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)


def calcular_graus(linhas, tol):
    """
    Calcula o grau de cada ponto (quantas linhas compartilham essa extremidade).

    Retorna: dicionário {chave_do_ponto: grau}
    """
    pontas = defaultdict(list)

    for i, l in enumerate(linhas):
        coords = list(l.coords)
        if len(coords) < 2:
            continue
        pontas[chave_ponto(coords[0], tol)].append((i, "start"))
        pontas[chave_ponto(coords[-1], tol)].append((i, "end"))

    graus = {k: len(v) for k, v in pontas.items()}
    return graus, pontas


def main():
    log("=" * 70)
    log("  REMOÇÃO DE GALHOS")
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
    log(f"Galho máximo: {cfg['galho_max_m']} m\n")

    # [1/3] Carrega
    log("[1/3] Carregando...")
    t0 = time.time()
    gdf = gpd.read_file(cfg["gpkg_entrada"])
    crs = gdf.crs.to_string() if gdf.crs else "EPSG:31983"
    linhas = extrair_linestrings(gdf)
    log(f"   Feições: {len(gdf):,}")
    log(f"   LineStrings: {len(linhas):,}\n")

    # [2/3] Analisa e remove
    log("[2/3] Analisando...")
    graus, pontas = calcular_graus(linhas, cfg["tolerancia_enc"])

    mantidas = []
    removidas = 0
    motivos = defaultdict(int)

    for i, l in enumerate(linhas):
        coords = list(l.coords)
        if len(coords) < 2:
            removidas += 1
            motivos["curta"] += 1
            continue

        p_ini = coords[0]
        p_fim = coords[-1]

        g_ini = graus.get(chave_ponto(p_ini, cfg["tolerancia_enc"]), 1)
        g_fim = graus.get(chave_ponto(p_fim, cfg["tolerancia_enc"]), 1)

        comp = l.length
        eh_galho = False

        # Regra 1: pendurado (uma ponta livre + uma em junção)
        if comp < cfg["galho_max_m"]:
            if (g_ini == 1 and g_fim >= 3) or (g_fim == 1 and g_ini >= 3):
                eh_galho = True
                motivos["galho_pendurado"] += 1

        # Regra 2: isolado (ambas pontas livres) e curto
        if not eh_galho and comp < cfg["galho_max_m"]:
            if g_ini == 1 and g_fim == 1:
                eh_galho = True
                motivos["isolado"] += 1

        if eh_galho:
            removidas += 1
            continue

        mantidas.append(l)

    log(f"   Linhas mantidas:  {len(mantidas):,}")
    log(f"   Linhas removidas: {removidas:,}")
    for k, v in motivos.items():
        log(f"     - {k}: {v:,}")
    log("")

    # [3/3] Salva
    log("[3/3] Salvando...")

    if not mantidas:
        raise SystemExit("[ERRO] Nada sobrou após remoção.")

    if os.path.exists(caminho_saida):
        try:
            os.remove(caminho_saida)
        except Exception:
            pass

    gdf_out = gpd.GeoDataFrame(geometry=mantidas, crs=crs)
    gdf_out.to_file(caminho_saida, driver="GPKG")

    comps = np.array([g.length for g in mantidas])
    log(f"\n{'=' * 70}")
    log(f"CONCLUÍDO em {(time.time() - t0):.1f}s")
    log(f"{'=' * 70}")
    log(f"Feições antes:   {len(linhas):,}")
    log(f"Feições depois:  {len(mantidas):,}")
    log(f"Removidas:       {removidas:,}")
    log(f"Comprimentos (m): min={comps.min():.1f}  "
        f"med={np.median(comps):.1f}  max={comps.max():.1f}")
    log(f"\nArquivo: {caminho_saida}")

    # Carrega no QGIS (se disponível)
    if cfg["carregar_qgis"]:
        try:
            from qgis.core import QgsProject, QgsVectorLayer
            nome_lyr = os.path.basename(caminho_saida).replace(".gpkg", "")
            lyr = QgsVectorLayer(caminho_saida, nome_lyr, "ogr")
            if lyr.isValid():
                QgsProject.instance().addMapLayer(lyr)
                log("Camada carregada no QGIS")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
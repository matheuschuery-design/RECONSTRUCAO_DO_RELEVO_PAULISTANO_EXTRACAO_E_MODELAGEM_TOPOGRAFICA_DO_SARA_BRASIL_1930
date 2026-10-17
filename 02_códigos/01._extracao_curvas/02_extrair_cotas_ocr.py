# -*- coding: utf-8 -*-
"""
================================================================================
 EXTRAÇÃO AUTOMÁTICA DE COTAS ALTIMÉTRICAS EM CARTAS TOPOGRÁFICAS POR OCR
================================================================================

 DESCRIÇÃO GERAL
 ---------------
 Este script extrai automaticamente os valores de cota altimétrica (números
 de altitude) presentes em cartas topográficas escaneadas, utilizando
 reconhecimento óptico de caracteres (OCR). É uma implementação genérica,
 aplicável a qualquer carta topográfica em que as cotas sejam representadas
 por algarismos em cor distinta do fundo.

 O script é parametrizável: todas as configurações (caminhos, parâmetros de
 OCR, faixas de cota aceitas, etc.) estão centralizadas no dicionário CONFIG.

 MODELO DE OCR UTILIZADO
 -----------------------
 Biblioteca: EasyOCR (Jaided AI)
 - Detector de texto: CRAFT (Character Region Awareness for Text detection)
 - Reconhecedor de texto: CRNN (Convolutional Recurrent Neural Network) + CTC
 - Idiomas suportados: Português (pt) e Inglês (en)
 - Modo de execução: CPU ou GPU (configurável)

 Referência do detector CRAFT:
 Baek, Y., Lee, B., Han, D., Yun, S., Lee, H. (2019).
 Character Region Awareness for Text Detection. CVPR 2019.

 Referência do reconhecedor CRNN:
 Shi, B., Bai, X., Yao, C. (2016).
 An End-to-End Trainable Neural Network for Image-based Sequence Recognition
 and Its Application to Scene Text Recognition. IEEE TPAMI.

 ESTRATÉGIA DE EXTRAÇÃO
 ----------------------
 O script adota uma estratégia robusta em seis etapas:

 1. DIVISÃO EM BLOCOS
    O raster é processado em blocos com sobreposição (overlap) para evitar
    cortar cotas na fronteira. Blocos sem conteúdo (sem cor de cota) são
    descartados antes do OCR, economizando tempo de processamento.

 2. PRÉ-PROCESSAMENTO
    Pixels pretos (linhas, hachuras, símbolos) são substituídos por cor de
    fundo para reduzir falsos positivos do OCR.

 3. DETECÇÃO DE COR
    Um filtro de cor (espaço HSV) verifica se o bloco contém a cor típica
    das cotas (por padrão, tons marrons). Blocos sem essa cor são pulados.

 4. OCR EM DOIS PASSES
    Cada bloco é submetido a duas passagens de OCR:
    - Passe "normal": parâmetros padrão (mais rápido e preciso)
    - Passe "sensível": parâmetros relaxados (captura textos difíceis)

 5. PÓS-PROCESSAMENTO
    Os textos reconhecidos passam por:
    - Correção de formato (ex.: "7.5O" → 750.0)
    - Validação de faixa altimétrica (ex.: 600–950 m)
    - Filtro de confiança mínima do OCR

 6. DEDUPLICAÇÃO ESPACIAL
    Cotas próximas (dentro de um raio configurável) com o mesmo valor são
    consolidadas em um único ponto (mantém o de maior confiança).

 PARÂMETROS ADAPTÁVEIS
 ---------------------
 Os seguintes parâmetros podem ser ajustados para diferentes cartas:

 - Faixa de cota (cota_min, cota_max): varia conforme a região geográfica.
 - Cor da cota (cor_h_min, cor_h_max, cor_s_min, cor_v_min): varia conforme
   a paleta da carta.
 - Tamanho do bloco (bloco) e overlap: ajustáveis conforme o raster.
 - Parâmetros do OCR (PASSES_OCR): ajustáveis para cartas mais ou menos
   legíveis.

 REQUISITOS
 ----------
 - Python 3.10+
 - EasyOCR         (`pip install easyocr`)
 - Rasterio        (`pip install rasterio`)
 - GeoPandas       (`pip install geopandas`)
 - Shapely, SciPy, NumPy, OpenCV

 INSTALAÇÃO EM DRIVE ALTERNATIVO
 -------------------------------
 Caso o espaço em disco no drive padrão seja limitado, as bibliotecas podem
 ser instaladas em outro drive:
   pip install easyocr --target D:\\pytorch_libs

 O script verifica automaticamente se essa pasta existe e a adiciona ao
 sys.path (comportamento controlado por CAMINHO_LIBS_EXTERNAS).

 TEMPO DE PROCESSAMENTO
 ----------------------
 O OCR é o gargalo principal. Estimativas aproximadas (CPU moderna):
 - Raster de 5.000 × 5.000 px:   ~30 min a 1 h
 - Raster de 10.000 × 10.000 px: ~2 h a 6 h
 - Raster de 20.000 × 20.000 px: ~8 h a 24 h

 Em GPU NVIDIA (CUDA), o tempo cai tipicamente 10× a 50×.

 O script salva progresso parcial periodicamente para permitir retomada.

 RESULTADO
 ---------
 Um shapefile de pontos (ou GeoPackage) contendo, para cada cota detectada:
 - Geometria (ponto) na posição original da cota na carta
 - Campo `valor`: valor altimétrico em metros
 - Campo `conf`: confiança do OCR (0.0 a 1.0)

 COMO USAR
 ---------
 1. Abra o terminal apropriado (OSGeo4W Shell ou similar)
 2. Edite o dicionário CONFIG abaixo com seus caminhos e parâmetros
 3. Execute:
       python 02_extrair_cotas_ocr.py

 4. Ao final, o log é aberto automaticamente no Notepad (Windows).

 LICENÇA
 -------
 Uso livre para fins acadêmicos e de pesquisa.

 AUTORIA
 -------
 [Seu nome]
 [Programa de Pós-Graduação]
 [Instituição]
 [Ano]
================================================================================
"""

import os
import sys
import time
import gc
import traceback

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
import cv2
import easyocr
import rasterio
from rasterio.windows import Window


# ==============================================================================
# CONFIGURAÇÕES — AJUSTE AQUI OS PARÂMETROS DO SEU PROJETO
# ==============================================================================
#
# Este bloco centraliza TODAS as configurações do script. Para usar com outra
# carta topográfica, basta ajustar estes valores. Não é necessário alterar
# nenhuma outra parte do código.
# ==============================================================================

CONFIG = {

    # --------------------------------------------------------------------------
    # CAMINHOS
    # --------------------------------------------------------------------------
    "raster_entrada": r"CAMINHO\PARA\sua_carta.tif",
    "pasta_saida":    r"CAMINHO\PARA\pasta\de\saida",
    "nome_saida":     "cotas_extraidas.shp",   # .shp ou .gpkg
    "nome_log":       "log_extrair_cotas.txt",

    # Caminho para bibliotecas instaladas em outro drive (opcional)
    # Defina como None se EasyOCR/PyTorch estiverem no ambiente padrão
    "caminho_libs_externas": None,  # ex.: r"D:\pytorch_libs"

    # --------------------------------------------------------------------------
    # PARÂMETROS DE PROCESSAMENTO EM BLOCOS
    # --------------------------------------------------------------------------
    "bloco":          2500,    # Tamanho do bloco (px). Aumentar = mais RAM
    "overlap":        250,     # Sobreposição entre blocos (px)
    "salvar_a_cada":  300,     # Salva progresso a cada N cotas

    # --------------------------------------------------------------------------
    # PARÂMETROS DE OCR
    # --------------------------------------------------------------------------
    "idiomas":        ["pt", "en"],
    "usar_gpu":       False,   # True se tiver GPU NVIDIA com CUDA
    "confianca_min":  0.2,     # Confiança mínima do OCR (0.0 a 1.0)

    # Faixa de cota aceita (em metros) — AJUSTAR conforme a região
    "cota_min":       600.0,
    "cota_max":       950.0,

    # --------------------------------------------------------------------------
    # PARÂMETROS DE DETECÇÃO DE COR (espaço HSV)
    # --------------------------------------------------------------------------
    # Define qual cor de texto é considerada "cota".
    # Padrão: tons marrons claros (típico de cartas antigas)
    #
    # Como calibrar:
    #   1. Abra a carta no QGIS
    #   2. Identifique visualmente a cor das cotas
    #   3. Ajuste os valores abaixo até o filtro detectar só as cotas
    "cor_h_min":      5,       # Hue mínimo (0–179)
    "cor_h_max":      30,      # Hue máximo (0–179)
    "cor_s_min":      50,      # Saturação mínima (0–255)
    "cor_v_min":      100,     # Brilho mínimo (0–255)
    "densidade_min":  0.005,   # Densidade mínima de cor para processar o bloco

    # --------------------------------------------------------------------------
    # PARÂMETROS DE DEDUPLICAÇÃO
    # --------------------------------------------------------------------------
    "raio_dedup":     15.0,    # Raio (m) para considerar duplicatas
}


# ==============================================================================
# PASSES DE OCR
# ==============================================================================
# Cada passe aplica o EasyOCR com parâmetros diferentes. Múltiplos passes
# aumentam a taxa de detecção, ao custo de mais tempo de processamento.
#
# - "normal": parâmetros padrão. Recomendado como primeiro passe.
# - "sensivel": parâmetros relaxados. Captura textos parcialmente ilegíveis,
#               inclinados ou com baixo contraste.
#
# Os parâmetros são documentados em:
# https://www.jaided.ai/easyocr/documentation/
# ==============================================================================

PASSES_OCR = [
    {
        "nome": "normal",
        "params": dict(
            detail=1,               # Retorna bounding box + texto + confiança
            paragraph=False,        # Não agrupa em parágrafos
            batch_size=8,           # Tamanho do batch (ajustar conforme RAM)
            text_threshold=0.4,     # Limiar de detecção de texto
            low_text=0.25,          # Limiar de detecção de caixa baixa
            link_threshold=0.25,    # Limiar de ligação entre caracteres
        ),
    },
    {
        "nome": "sensivel",
        "params": dict(
            detail=1,
            paragraph=True,
            batch_size=8,
            text_threshold=0.3,     # Mais permissivo
            low_text=0.2,           # Mais permissivo
            link_threshold=0.2,     # Mais permissivo
            mag_ratio=1.3,          # Amplia a imagem antes do OCR
        ),
    },
]


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

class Logger:
    """
    Sistema de log duplo: escreve simultaneamente no terminal e em arquivo.
    Útil para acompanhar o progresso em tempo real e ter registro completo.
    """
    def __init__(self, caminho):
        self.caminho = caminho
        self.file = open(caminho, "w", encoding="utf-8")

    def log(self, msg=""):
        print(msg, flush=True)
        self.file.write(str(msg) + "\n")
        self.file.flush()

    def fechar(self):
        self.file.close()


def adicionar_caminho_libs(caminho):
    """
    Adiciona um caminho externo ao sys.path, permitindo importar bibliotecas
    instaladas em outro drive.
    """
    if caminho and os.path.isdir(caminho):
        sys.path.insert(0, caminho)


def corrigir_formato_cota(texto):
    """
    Corrige o formato dos valores numéricos lidos pelo OCR.

    Estratégia:
    - Remove caracteres não numéricos (exceto ponto)
    - Aceita valores de 3 dígitos como inteiros (ex.: "750" → 750.0)
    - Aceita valores de 4 dígitos assumindo o último como decimal
      (ex.: "7505" → 750.5)

    Parâmetros:
    - texto: string lida pelo OCR

    Retorna: float ou None (se não for possível corrigir)
    """
    # Remove tudo que não é dígito ou ponto
    t = ''.join(c for c in texto if c.isdigit() or c == '.')

    # Caso 1: já tem ponto decimal
    if '.' in t:
        partes = t.split('.')
        if len(partes) == 2:
            inteiro = ''.join(c for c in partes[0] if c.isdigit())
            frac = ''.join(c for c in partes[1] if c.isdigit())
            if len(inteiro) == 3 and len(frac) >= 1:
                try:
                    return float(f"{inteiro}.{frac[0]}")
                except ValueError:
                    pass

    # Caso 2: sem ponto — conta dígitos
    td = ''.join(c for c in t if c.isdigit())

    if len(td) == 3:
        return float(td)

    if len(td) == 4:
        return float(f"{td[:3]}.{td[3]}")

    return None


def filtrar_pixels_pretos(img_rgb):
    """
    Substitui pixels pretos por cor de fundo.

    Pixels pretos geralmente correspondem a linhas, hachuras ou símbolos da
    carta, que podem ser erroneamente interpretados como texto pelo OCR.
    Substituí-los por uma cor neutra melhora a precisão.

    Parâmetros:
    - img_rgb: array numpy (H, W, 3) em RGB

    Retorna: array numpy (H, W, 3) modificado
    """
    img_mod = img_rgb.copy()

    r = img_mod[:, :, 0].astype(int)
    g = img_mod[:, :, 1].astype(int)
    b = img_mod[:, :, 2].astype(int)

    # Define "preto": todos os canais abaixo de 120 e semelhantes
    mascara_preto = (
        (r < 120) & (g < 120) & (b < 120) &
        (np.abs(r - g) < 25) & (np.abs(r - b) < 25)
    )

    # Dilata a máscara para cobrir bordas
    mascara_dil = binary_dilation(mascara_preto, iterations=2)

    # Substitui por cor de fundo (bege claro neutro)
    img_mod[mascara_dil] = [240, 240, 235]

    return img_mod


def bloco_tem_cota(img_mod, cfg):
    """
    Verifica se o bloco contém pixels com cor típica de cota.

    Essa verificação economiza muito tempo, pois blocos sem essa cor
    (ex.: áreas vazias, florestas, corpos d'água) são pulados sem OCR.

    Parâmetros:
    - img_mod: array numpy (H, W, 3) em RGB
    - cfg: dicionário CONFIG

    Retorna: True se o bloco tem densidade de cor suficiente, False caso contrário
    """
    img_hsv = cv2.cvtColor(img_mod, cv2.COLOR_RGB2HSV)

    mask = (
        (img_hsv[:, :, 0] >= cfg["cor_h_min"]) &
        (img_hsv[:, :, 0] <= cfg["cor_h_max"]) &
        (img_hsv[:, :, 1] > cfg["cor_s_min"]) &
        (img_hsv[:, :, 2] > cfg["cor_v_min"])
    )

    densidade = mask.sum() / mask.size
    return densidade >= cfg["densidade_min"]


def deduplicar_pontos(pontos, raio):
    """
    Remove cotas duplicadas que estão muito próximas.

    Como o raster é processado em blocos com overlap, uma mesma cota pode
    ser detectada múltiplas vezes. Essa função consolida as detecções:

    - Pontos dentro do raio especificado com o mesmo valor são agrupados
    - Mantém apenas o de maior confiança

    Parâmetros:
    - pontos: lista de dicionários com chaves 'x', 'y', 'valor', 'conf'
    - raio: raio de deduplicação em metros

    Retorna: lista de pontos únicos
    """
    if not pontos:
        return []

    coords = np.array([[p["x"], p["y"]] for p in pontos])
    tree = cKDTree(coords)

    usados = set()
    finais = []

    for i, p in enumerate(pontos):
        if i in usados:
            continue

        vizinhos = tree.query_ball_point(coords[i], raio)
        mesmo_valor = [
            j for j in vizinhos
            if abs(pontos[j]["valor"] - p["valor"]) < 0.05
        ]

        if len(mesmo_valor) > 1:
            melhor = max(mesmo_valor, key=lambda k: pontos[k]["conf"])
            finais.append(pontos[melhor])
            for k in mesmo_valor:
                usados.add(k)
        else:
            finais.append(p)
            usados.add(i)

    return finais


def salvar_pontos(pontos, caminho, crs):
    """
    Salva a lista de pontos como shapefile ou GeoPackage.

    Parâmetros:
    - pontos: lista de dicionários
    - caminho: caminho completo do arquivo de saída (.shp ou .gpkg)
    - crs: sistema de referência de coordenadas
    """
    if not pontos:
        return

    geometrias = [Point(p["x"], p["y"]) for p in pontos]
    gdf = gpd.GeoDataFrame({
        "valor": [p["valor"] for p in pontos],
        "conf":  [p["conf"] for p in pontos],
    }, geometry=geometrias, crs=crs)

    gdf.to_file(caminho)


# ==============================================================================
# PROCESSAMENTO PRINCIPAL
# ==============================================================================

def processar_raster(cfg, reader, log):
    """
    Executa a extração de cotas em um raster completo.

    Parâmetros:
    - cfg: dicionário CONFIG
    - reader: instância do EasyOCR carregada
    - log: função de log

    Retorna: lista de pontos extraídos
    """
    # Abre o raster
    with rasterio.open(cfg["raster_entrada"]) as src:
        W, H = src.width, src.height
        transform = src.transform
        crs = src.crs

    log(f"Raster de entrada: {W} × {H} px")
    log(f"CRS: {crs}\n")

    pontos = []
    n_blocos = 0
    n_blocos_ocr = 0
    n_vazios = 0
    n_aceitos = 0

    t0 = time.time()
    step = cfg["bloco"] - cfg["overlap"]
    total = ((W // step) + 1) * ((H // step) + 1)

    log(f">>> Processando ~{total} blocos...\n")

    caminho_saida = os.path.join(cfg["pasta_saida"], cfg["nome_saida"])

    with rasterio.open(cfg["raster_entrada"]) as src:
        for y0 in range(0, H - cfg["bloco"], step):
            for x0 in range(0, W - cfg["bloco"], step):
                n_blocos += 1

                # Lê bloco
                win = Window(x0, y0, cfg["bloco"], cfg["bloco"])
                try:
                    img_rgb = src.read(window=win)
                except Exception:
                    continue

                if (img_rgb is None or
                        img_rgb.shape[1] < cfg["bloco"] or
                        img_rgb.shape[2] < cfg["bloco"]):
                    continue

                img_mod = img_rgb.transpose(1, 2, 0).copy()

                # Pré-processamento
                img_mod = filtrar_pixels_pretos(img_mod)

                # Verifica se tem cota
                if not bloco_tem_cota(img_mod, cfg):
                    n_vazios += 1
                    del img_rgb, img_mod
                    gc.collect()
                    continue

                n_blocos_ocr += 1

                # Aplica os passes de OCR
                for passe in PASSES_OCR:
                    try:
                        resultados = reader.readtext(img_mod, **passe["params"])
                    except Exception:
                        continue

                    for item in resultados:
                        if len(item) == 3:
                            bbox, texto, conf = item
                        elif len(item) == 2:
                            bbox, texto = item
                            conf = 0.5
                        else:
                            continue

                        if conf < cfg["confianca_min"]:
                            continue

                        valor = corrigir_formato_cota(texto)
                        if valor is None:
                            continue

                        if not (cfg["cota_min"] <= valor <= cfg["cota_max"]):
                            continue

                        # Centro do bbox em pixels
                        xs = [int(pt[0]) for pt in bbox]
                        ys = [int(pt[1]) for pt in bbox]
                        bx1, bx2 = min(xs), max(xs)
                        by1, by2 = min(ys), max(ys)

                        cx = x0 + (bx1 + bx2) / 2.0
                        cy = y0 + (by1 + by2) / 2.0

                        # Converte para coordenadas do CRS original
                        lon, lat = transform * (cx, cy)

                        pontos.append({
                            "x": lon,
                            "y": lat,
                            "valor": valor,
                            "conf": float(conf),
                            "texto": str(texto),
                        })
                        n_aceitos += 1

                del img_rgb, img_mod
                try:
                    del resultados
                except NameError:
                    pass
                gc.collect()

                # Progresso
                if n_blocos_ocr % 3 == 0:
                    dt = time.time() - t0
                    vel = n_blocos / dt if dt > 0 else 0
                    resta = (total - n_blocos) / vel / 60 if vel > 0 else 0
                    log(f"  bloco {n_blocos}/{total} | OCR: {n_blocos_ocr} | "
                        f"brutos: {n_aceitos} | {dt/60:.1f} min | resta ~{resta:.0f} min")

                # Salva progresso parcial
                if n_aceitos > 0 and n_aceitos % cfg["salvar_a_cada"] == 0:
                    try:
                        salvar_pontos(pontos, caminho_saida, crs)
                        log(f"  progresso salvo: {n_aceitos} cotas")
                    except Exception as e:
                        log(f"  erro salvando parcial: {e}")

    log(f"\n{'=' * 70}")
    log(f"Resumo da extração bruta")
    log(f"{'=' * 70}")
    log(f"  Cotas detectadas:  {n_aceitos:,}")
    log(f"  Blocos com OCR:    {n_blocos_ocr}")
    log(f"  Blocos vazios:     {n_vazios}")
    log(f"  Tempo total:       {(time.time() - t0) / 60:.1f} min")
    log(f"{'=' * 70}\n")

    return pontos, crs


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    # Adiciona caminho externo de bibliotecas (se configurado)
    adicionar_caminho_libs(CONFIG["caminho_libs_externas"])

    # Configura logger
    os.makedirs(CONFIG["pasta_saida"], exist_ok=True)
    caminho_log = os.path.join(CONFIG["pasta_saida"], CONFIG["nome_log"])
    logger = Logger(caminho_log)
    log = logger.log

    try:
        log("=" * 70)
        log("  EXTRAÇÃO DE COTAS ALTIMÉTRICAS — OCR (EasyOCR)")
        log("=" * 70)
        log("")

        # Validações
        if not os.path.isfile(CONFIG["raster_entrada"]):
            raise SystemExit(f"[ERRO] Raster não encontrado: {CONFIG['raster_entrada']}")

        # Carrega EasyOCR
        log("Carregando EasyOCR...")
        reader = easyocr.Reader(
            CONFIG["idiomas"],
            gpu=CONFIG["usar_gpu"],
            verbose=False,
        )
        log("   OK\n")

        # Processa
        pontos, crs = processar_raster(CONFIG, reader, log)

        # Deduplica
        log(f">>> Deduplicando (raio {CONFIG['raio_dedup']} m)...")
        finais = deduplicar_pontos(pontos, CONFIG["raio_dedup"])
        log(f"   Após dedup: {len(finais):,} cotas únicas\n")

        # Salva resultado final
        caminho_saida = os.path.join(CONFIG["pasta_saida"], CONFIG["nome_saida"])
        salvar_pontos(finais, caminho_saida, crs)
        log(f"OK: {caminho_saida}")

    except Exception:
        log(f"\nERRO:\n{traceback.format_exc()}")

    finally:
        logger.fechar()
        print(f"\nLog salvo em: {caminho_log}")
        if os.name == "nt":  # Windows
            import subprocess
            subprocess.Popen(["notepad.exe", caminho_log])


if __name__ == "__main__":
    main()
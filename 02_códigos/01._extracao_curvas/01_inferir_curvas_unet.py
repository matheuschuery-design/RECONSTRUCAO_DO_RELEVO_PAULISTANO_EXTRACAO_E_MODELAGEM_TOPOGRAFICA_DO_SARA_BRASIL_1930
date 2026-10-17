# -*- coding: utf-8 -*-
"""
================================================================================
 INFERÊNCIA DE CURVAS DE NÍVEL EM CARTAS TOPOGRÁFICAS COM REDE U-NET
================================================================================

 DESCRIÇÃO GERAL
 ---------------
 Este script aplica um modelo de rede neural convolucional do tipo U-Net
 pré-treinado sobre uma carta topográfica escaneada (em formato raster RGB),
 extraindo automaticamente as curvas de nível como uma máscara binária.

 É uma implementação genérica, aplicável a qualquer carta topográfica em
 que as curvas de nível sejam visualmente distintas do fundo. Todos os
 parâmetros são configuráveis via dicionário CONFIG.

 MODELO DE SEGMENTAÇÃO UTILIZADO
 -------------------------------
 Arquitetura: U-Net (Ronneberger et al., 2015)
 - Encoder: 3 blocos convolucionais (32 → 64 → 128 canais)
 - Bottleneck: 1 bloco (256 canais)
 - Decoder: 3 blocos com upsampling por transposição convolucional
 - Skip connections: entre encoder e decoder (preserva detalhe espacial)
 - Saída: 1 canal (segmentação binária via sigmoide)
 - Batch Normalization em todas as camadas convolucionais
 - Ativação: ReLU nas camadas internas, sigmoide na saída

 DADOS DE TREINAMENTO (do modelo incluso)
 ----------------------------------------
 O modelo padrão foi treinado com:
 - Entrada: patches RGB 256×256 px recortados de cartas topográficas
 - Alvo: máscaras binárias anotadas (curva vs. fundo)
 - Aprendizado supervisionado pixel a pixel

 Referência do modelo original:
 Ronneberger, O., Fischer, P., Brox, T. (2015).
 U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

 ESTRATÉGIA DE INFERÊNCIA
 ------------------------
 O raster é processado em tiles com overlap para evitar artefatos nas bordas:

 1. O raster é dividido em tiles de tamanho configurável (default 1024×1024 px)
 2. Cada tile é lido com uma margem de overlap (default 64 px)
 3. Cada tile é subdividido em patches de 256×256 px (tamanho de treino)
 4. Cada patch é processado individualmente pela U-Net
 5. Apenas o núcleo de cada tile é gravado (evita duplicação na overlap)
 6. Checkpoint é salvo a cada N tiles (permite retomada automática)

 ADAPTAÇÃO A OUTROS MODELOS
 --------------------------
 Para usar com outro modelo treinado:
 1. Ajustar o caminho do arquivo `.pt` em CONFIG
 2. Se a arquitetura for diferente, adaptar as classes BlocoConvolucional
    e UNet para corresponder à topologia original
 3. Se as bandas de entrada mudarem, ajustar `nn.Conv2d(3, ...)` na primeira
    camada convolucional
 4. Ajustar `patch` em CONFIG se o modelo exigir outro tamanho de entrada

 REQUISITOS
 ----------
 - Python 3.10+
 - PyTorch (`pip install torch`)
 - OpenCV  (`pip install opencv-python`)
 - GDAL    (via OSGeo4W)
 - NumPy

 INSTALAÇÃO EM DRIVE ALTERNATIVO
 -------------------------------
 Caso o espaço em disco no drive padrão seja limitado:
   pip install torch --target D:\\pytorch_libs

 O script verifica automaticamente se essa pasta existe e a adiciona ao
 sys.path (controlado por CAMINHO_LIBS_EXTERNAS em CONFIG).

 TEMPO DE PROCESSAMENTO
 ----------------------
 O tempo depende do tamanho do raster e da CPU/GPU disponível:

 - CPU moderna: ~1–5 tiles/s (para tiles de 1024×1024 px)
 - GPU NVIDIA:   ~10–50× mais rápido

 Estimativas para raster de 20.000×20.000 px em CPU: ~1–3 h.

 COMO USAR
 ---------
 1. Abra o OSGeo4W Shell (ou terminal similar)
 2. Edite o dicionário CONFIG abaixo com seus caminhos
 3. Execute:
       python 01_inferir_curvas_unet.py

 4. Ao final, o resultado é um GeoTIFF binário (0 = fundo, 255 = curva)

 COMO RETOMAR SE INTERROMPIDO
 ----------------------------
 O script salva checkpoints automaticamente. Se for interrompido, basta
 rodar novamente — ele retoma do último tile processado. Para reiniciar do
 zero, apague o arquivo `inferencia_ckpt.pkl` na pasta de saída.

 RESULTADO
 ---------
 Um arquivo GeoTIFF georreferenciado (mesmo CRS do raster de entrada),
 contendo uma máscara binária:
 - 0   = pixel de fundo (não é curva)
 - 255 = pixel de curva detectada

 PRÓXIMOS PASSOS (após este script)
 ----------------------------------
 O raster binário gerado é insumo para:
 1. Vetorização (`gdal_polygonize.py` ou QGIS)
 2. Limpeza topológica (`v.clean` no GRASS GIS)
 3. Atribuição de cotas (por OCR ou proximidade espacial)

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
import pickle

import numpy as np
import cv2
import torch
import torch.nn as nn
from osgeo import gdal


# ==============================================================================
# CONFIGURAÇÕES — AJUSTE AQUI OS PARÂMETROS DO SEU PROJETO
# ==============================================================================
#
# Este bloco centraliza TODAS as configurações do script. Para usar com
# outro modelo ou outra carta, basta ajustar estes valores.
# ==============================================================================

CONFIG = {

    # --------------------------------------------------------------------------
    # CAMINHOS
    # --------------------------------------------------------------------------
    "modelo":         r"CAMINHO\PARA\modelo_treinado.pt",
    "raster_entrada": r"CAMINHO\PARA\carta_topo.tif",
    "pasta_saida":    r"CAMINHO\PARA\pasta\de\saida",
    "nome_saida":     "curvas_brutas.tif",

    # Caminho para bibliotecas instaladas em outro drive (opcional)
    # Defina como None se PyTorch já estiver no ambiente padrão
    "caminho_libs_externas": None,  # ex.: r"D:\pytorch_libs"

    # --------------------------------------------------------------------------
    # PARÂMETROS DE INFERÊNCIA
    # --------------------------------------------------------------------------
    "tile":             1024,   # Tamanho do tile de processamento (px)
    "patch":            256,    # Tamanho do patch de entrada da rede (px)
    "overlap":          64,     # Sobreposição entre tiles (px)
    "threshold":        0.5,    # Limiar de probabilidade (0.0 a 1.0)
    "num_threads":      4,      # Threads de CPU para PyTorch
    "checkpoint_a_cada": 200,   # Salva progresso a cada N tiles
}


# ==============================================================================
# ARQUITETURA U-NET
# ==============================================================================
# A arquitetura abaixo é a mesma utilizada no treinamento do modelo fornecido.
# Se o usuário treinar um modelo com arquitetura diferente, deve adaptar
# estas classes para corresponder à topologia original.
#
# Diagrama:
#
#   ENTRADA (3 canais RGB)
#          │
#          ▼
#   ┌──────────────┐
#   │  Encoder 1   │ 32 ch
#   └──────────────┘
#          │ ─────────────────────────┐ (skip connection)
#          ▼                          │
#   ┌──────────────┐                  │
#   │  Encoder 2   │ 64 ch            │
#   └──────────────┘                  │
#          │ ───────────────────┐     │
#          ▼                    │     │
#   ┌──────────────┐            │     │
#   │  Encoder 3   │ 128 ch     │     │
#   └──────────────┘            │     │
#          │                    │     │
#          ▼                    │     │
#   ┌──────────────┐            │     │
#   │  Bottleneck  │ 256 ch     │     │
#   └──────────────┘            │     │
#          │                    │     │
#          ▼                    │     │
#   ┌──────────────┐            │     │
#   │  Decoder 3   │ ◄──────────┘     │
#   └──────────────┘                  │
#          │                          │
#          ▼                          │
#   ┌──────────────┐                  │
#   │  Decoder 2   │ ◄────────────────┘
#   └──────────────┘
#          │
#          ▼
#   ┌──────────────┐
#   │  Decoder 1   │
#   └──────────────┘
#          │
#          ▼
#   SAÍDA (1 canal — probabilidade)
# ==============================================================================

class BlocoConvolucional(nn.Module):
    """
    Bloco duplo de convolução 3×3 + BatchNorm + ReLU.
    Unidade básica do encoder e decoder da U-Net.
    """
    def __init__(self, entrada, saida):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(entrada, saida, 3, padding=1),
            nn.BatchNorm2d(saida),
            nn.ReLU(inplace=True),
            nn.Conv2d(saida, saida, 3, padding=1),
            nn.BatchNorm2d(saida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    U-Net com 3 níveis de encoder/decoder.

    Canais por nível: 32 → 64 → 128 → 256 (bottleneck)
    """
    def __init__(self, canais=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = canais

        # Encoder
        self.enc1 = BlocoConvolucional(3, c1)
        self.enc2 = BlocoConvolucional(c1, c2)
        self.enc3 = BlocoConvolucional(c2, c3)

        # Bottleneck
        self.bottleneck = BlocoConvolucional(c3, c3 * 2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(c3 * 2, c3, 2, 2)
        self.dec3 = BlocoConvolucional(c3 * 2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = BlocoConvolucional(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = BlocoConvolucional(c1 * 2, c1)

        # Saída
        self.out = nn.Conv2d(c1, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        d1 = self.enc1(x)
        d2 = self.enc2(self.pool(d1))
        d3 = self.enc3(self.pool(d2))

        # Bottleneck
        b = self.bottleneck(self.pool(d3))

        # Decoder com skip connections
        u3 = self.dec3(torch.cat([self.up3(b), d3], 1))
        u2 = self.dec2(torch.cat([self.up2(u3), d2], 1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], 1))

        return self.out(u1)


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def log(msg=""):
    """Imprime com flush imediato (acompanhamento em tempo real)."""
    print(msg, flush=True)


def adicionar_caminho_libs(caminho):
    """Adiciona um caminho externo ao sys.path (para libs em outro drive)."""
    if caminho and os.path.isdir(caminho):
        sys.path.insert(0, caminho)


def validar_config(cfg):
    """Verifica se os arquivos e pastas configurados existem."""
    if not os.path.isfile(cfg["modelo"]):
        raise SystemExit(f"[ERRO] Modelo não encontrado: {cfg['modelo']}")
    if not os.path.isfile(cfg["raster_entrada"]):
        raise SystemExit(f"[ERRO] Raster não encontrado: {cfg['raster_entrada']}")
    if not os.path.isdir(cfg["pasta_saida"]):
        raise SystemExit(f"[ERRO] Pasta de saída não existe: {cfg['pasta_saida']}")


def carregar_modelo(caminho, num_threads):
    """
    Carrega o modelo U-Net em modo de avaliação.

    Parâmetros:
    - caminho: caminho do arquivo .pt salvo por torch.save
    - num_threads: número de threads de CPU para PyTorch

    Retorna: instância do modelo carregado
    """
    modelo = UNet().to("cpu")
    modelo.load_state_dict(torch.load(caminho, map_location="cpu"))
    modelo.eval()
    torch.set_num_threads(num_threads)
    return modelo


def predizer_patchs(modelo, img_bgr, patch_size, threshold):
    """
    Aplica o modelo em uma imagem, dividindo em patches de tamanho fixo.

    Parâmetros:
    - modelo: instância UNet carregada
    - img_bgr: array numpy (H, W, 3) em BGR (formato do OpenCV)
    - patch_size: tamanho do patch (deve corresponder ao treino)
    - threshold: limiar de probabilidade (0.0 a 1.0)

    Retorna: array numpy (H, W) com valores 0 (fundo) ou 255 (curva)
    """
    h, w = img_bgr.shape[:2]
    saida = np.zeros((h, w), dtype=np.uint8)

    with torch.no_grad():
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = img_bgr[y:y+patch_size, x:x+patch_size]

                # Pula patches incompletos (nas bordas)
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    continue

                # Converte BGR → RGB e normaliza para [0, 1]
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                # Converte para tensor (1, C, H, W)
                tensor = torch.from_numpy(np.transpose(patch_rgb, (2, 0, 1))).unsqueeze(0)

                # Inferência
                prob = torch.sigmoid(modelo(tensor))[0, 0].numpy()

                # Binariza pelo threshold
                saida[y:y+patch_size, x:x+patch_size] = (prob > threshold).astype(np.uint8) * 255

    return saida


def inferir_raster(modelo, raster_path, saida_path, cfg):
    """
    Executa a inferência em um raster completo, por tiles.

    Suporta retomada automática se interrompido.

    Retorna: tempo de processamento em segundos
    """
    ds = gdal.Open(raster_path)
    if ds is None:
        raise SystemExit(f"[ERRO] GDAL não conseguiu abrir: {raster_path}")

    W, H = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    log(f"   Dimensões: {W} × {H} px")
    log(f"   Bandas: {ds.RasterCount}")

    # Cria raster de saída
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        saida_path, W, H, 1, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)

    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    # Checkpoint
    ckpt_path = os.path.join(cfg["pasta_saida"], "inferencia_ckpt.pkl")
    tile_inicial = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            tile_inicial = pickle.load(f)["tile"]
        log(f"   [RETOMANDO] do tile {tile_inicial}")

    # Lista de tiles
    tile_size = cfg["tile"]
    overlap = cfg["overlap"]
    step = tile_size - overlap
    tiles = [(x, y) for y in range(0, H, step) for x in range(0, W, step)]
    log(f"   Total de tiles: {len(tiles)}")

    t0 = time.time()
    for k in range(tile_inicial, len(tiles)):
        x, y = tiles[k]

        # Lê região com margem
        rx = max(0, x - overlap)
        ry = max(0, y - overlap)
        rw = min(tile_size + 2 * overlap, W - rx)
        rh = min(tile_size + 2 * overlap, H - ry)

        r = ds.GetRasterBand(1).ReadAsArray(rx, ry, rw, rh)
        g = ds.GetRasterBand(2).ReadAsArray(rx, ry, rw, rh)
        b = ds.GetRasterBand(3).ReadAsArray(rx, ry, rw, rh)
        if r is None:
            continue

        # Monta imagem BGR
        img = np.dstack((b, g, r)).astype(np.uint8)

        # Prediz
        pred = predizer_patchs(modelo, img, cfg["patch"], cfg["threshold"])

        # Recorta o núcleo (sem overlap) e escreve
        cx1, cy1 = x - rx, y - ry
        cw = min(tile_size, W - x)
        ch = min(tile_size, H - y)
        core = pred[cy1:cy1+ch, cx1:cx1+cw]
        band.WriteArray((core > 0).astype(np.uint8), x, y)

        # Progresso
        if (k + 1) % 50 == 0 or k == len(tiles) - 1:
            dt = (time.time() - t0) / 60
            feitos = k + 1 - tile_inicial
            resta = dt / max(1, feitos) * (len(tiles) - tile_inicial - feitos)
            log(f"   {k+1}/{len(tiles)} | {dt:.1f} min | resta ~{resta:.1f} min")

        # Checkpoint
        if (k + 1) % cfg["checkpoint_a_cada"] == 0:
            with open(ckpt_path, "wb") as f:
                pickle.dump({"tile": k + 1}, f)

    # Finaliza
    band.FlushCache()
    out_ds = None
    ds = None

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return time.time() - t0


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    # Adiciona caminho externo de bibliotecas
    adicionar_caminho_libs(CONFIG["caminho_libs_externas"])

    log("=" * 70)
    log("  INFERÊNCIA DE CURVAS DE NÍVEL — U-NET")
    log("=" * 70)
    log("")
    log("  Modelo:  U-Net (3 níveis, 32→64→128 canais)")
    log("  Entrada: carta topográfica em raster RGB")
    log("  Saída:   máscara binária (0 = fundo, 255 = curva)")
    log("")

    validar_config(CONFIG)

    saida_path = os.path.join(CONFIG["pasta_saida"], CONFIG["nome_saida"])

    log(f"  Modelo:    {os.path.basename(CONFIG['modelo'])}")
    log(f"  Raster:    {os.path.basename(CONFIG['raster_entrada'])}")
    log(f"  Saída:     {saida_path}")
    log(f"  Tile:      {CONFIG['tile']} px | Patch: {CONFIG['patch']} px | "
        f"Overlap: {CONFIG['overlap']} px")
    log(f"  Threshold: {CONFIG['threshold']}")
    log("")
    log("-" * 70)

    log("\n[1/3] Carregando modelo...")
    modelo = carregar_modelo(CONFIG["modelo"], CONFIG["num_threads"])
    log("   OK — modelo carregado em modo eval")

    log("\n[2/3] Executando inferência...")
    log(f"   (checkpoint automático a cada {CONFIG['checkpoint_a_cada']} tiles)\n")

    tempo = inferir_raster(modelo, CONFIG["raster_entrada"], saida_path, CONFIG)

    log("\n[3/3] Finalizado")
    log("-" * 70)
    log(f"\n   Arquivo:  {saida_path}")
    log(f"   Tamanho:  {os.path.getsize(saida_path) / 1e6:.1f} MB")
    log(f"   Tempo:    {tempo / 60:.1f} min")
    log("\n" + "=" * 70)
    log("  PRÓXIMOS PASSOS")
    log("=" * 70)
    log("  1. Abrir no QGIS")
    log("  2. Raster → Extração → Contornos (ou gdal_polygonize.py)")
    log("  3. Limpar feições com v.clean ou QGIS")
    log("  4. Atribuir cotas por OCR ou proximidade espacial")
    log("")


if __name__ == "__main__":
    main()
# RECONSTRUÇÃO DO RELEVO PAULISTANO EXTRAÇÃO E MODELAGEM TOPOGRÁFICA DO SARA BRASIL (1930)
The project reconstructs São Paulo’s historical terrain using the 1930 Sara Brasil survey, combining machine learning and geoprocessing to extract contour lines and generate a Digital Terrain Model (DTM) for analyzing topographic changes over time.

O projeto busca reconstruir o relevo histórico de São Paulo a partir do levantamento cartográfico Sara Brasil, de 1930, por meio de técnicas de geoprocessamento, aprendizado de máquina e automação computacional. O trabalho envolve a extração e vetorização de curvas de nível, o reconhecimento automatizado de cotas altimétricas e a geração de um Modelo Digital do Terreno (MDT). A base produzida constitui uma referência para investigar as transformações da topografia paulistana ao longo do tempo, a partir da comparação entre o relevo histórico e o atual.Todo o código-fonte utilizado neste projeto está sendo disponibilizado publicamente no GitHub.

Pesquisa de Mestrado
Faculdade de Arquitetura e Urbanismo e de Design
Universidade de São Paulo (FAU-USP)

Autor: Matheus Chuery Pinto
Período: 2025–2027
Contato: matheuschuery@usp.br

O presente trabalho foi realizado com apoio do Conselho Nacional de
Desenvolvimento Científico e Tecnológico (CNPq – Brasil) – Processo
nº 133553/2025-6.

Todo o conteúdo deste projeto é de livre acesso, desde que citado o autor.

The project aims to reconstruct the historical terrain of São Paulo using the 1930 Sara Brasil cartographic survey through geoprocessing, machine learning, and computational automation techniques. The workflow involves the extraction and vectorization of contour lines, the automated recognition of elevation values, and the generation of a Digital Terrain Model (DTM). The resulting dataset provides a foundation for investigating changes in São Paulo’s topography over time by comparing historical and present-day terrain models. All source code used in this project is being made publicly available on GitHub.

1. Arquivos e diretórios

A estrutura abaixo apresenta a organização dos dados originais, códigos, arquivos intermediários e produtos finais do projeto.

00_PROJETO RECONSTRUÇÃO DO RELEVO PAULISTANO/
├── 00_ESTRUTURA_PROJETO_RELEVO_PAULISTANO.txt
├── MDT_SARA_BRASIL_1930_10m.tif
├── Sara_brasil_1930_modelo_tridimencional.glb
├── 01_dados_originais/
│   ├── raster_mapa_1897_georeferenciado.tif
│   └── raster_sara_1930_curvas_5m.tif
├── 02_codigos/
│   ├── 01_extracao_curvas/
│   │   ├── pipeline.txt
│   │   ├── 01_inferir_curvas_unet.py
│   │   ├── 02_extrair_cotas_ocr.py
│   │   └── modelo_unet/
│   │       ├── unet_curvas_nivel.pt
│   │       └── unet_curvas_nivel_ep010.pt
│   ├── 02_processamento_curvas/
│   │   ├── pipeline.txt
│   │   ├── 03_limpar_raster_curvas.py
│   │   ├── 04_esqueletizar_vetorizar.py
│   │   ├── 05_conectar_gaps.py
│   │   ├── 06_snap_merge.py
│   │   ├── 07_remover_galhos.py
│   │   └── 08_limpar_topologia_grass.py
│   └── 03_visualizacao_3d/
│       └── 10_gerar_4_modelos.py
├── 03_desenvolvimento/
├── 04_camadas_finais/
│   ├── vetoriais/
│   └── rasters/
├── 05_mapas_publicacao/
└── 06_modelos_3d/

2. Pipeline de processamento

Dados originais
      │
      ▼
Extração automática das curvas (U-Net)
      │
      ▼
Limpeza do raster
      │
      ▼
Esqueletização e vetorização
      │
      ▼
Conexão de segmentos e correção topológica
      │
      ▼
Camadas vetoriais finais
      │
      ▼
Interpolação RST no GRASS GIS
      │
      ▼
Modelo Digital do Terreno (MDT)
      │
      ├── Hillshade e produtos cartográficos
      ├── Análises morfométricas
      └── Modelos tridimensionais

3. Notas

Os arquivos intermediários documentam as etapas de processamento e desenvolvimento.

Os vetores são armazenados em GeoPackage (.gpkg).

Os rasters são armazenados em GeoTIFF (.tif).

Os modelos tridimensionais estão disponíveis em formatos OBJ, MTL e GLB, conforme o produto.

Arquivos auxiliares gerados pelo QGIS podem ser ignorados.

4. Licença e uso

O conteúdo deste projeto é disponibilizado para fins acadêmicos, de pesquisa e educacionais, desde que o autor seja devidamente citado. Consulte as condições da licença do repositório antes de redistribuir ou utilizar os materiais.

Forma sugerida de citação

CHUERY PINTO, Matheus. Reconstrução do Relevo Paulistano: extração automática de curvas de nível do mapeamento Sara Brasil (1930) e geração de Modelos Digitais de Terreno. Dissertação (Mestrado em Arquitetura e Urbanismo) – Faculdade de Arquitetura, Urbanismo e de Design, Universidade de São Paulo, São Paulo, 2027.

5. Agradecimentos

O presente trabalho foi realizado com apoio do Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq – Brasil) – Processo nº 133553/2025-6.

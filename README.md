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

================================================================================
 PROJETO: RECONSTRUÇÃO DO RELEVO PAULISTANO
 Estrutura de organização de arquivos
================================================================================

--------------------------------------------------------------------------------
 ARQUIVOS RAIZ
--------------------------------------------------------------------------------

00_PROJETO RECONSTRUÇÃO DO RELEVO PAULISTANO/
│
├── 00_ESTRUTURA_PROJETO_RELEVO_PAULISTANO.txt
│       Este arquivo. Documenta a organização do projeto.
│
├── MDT_SARA_BRASIL_1930_10m.tif
│       Modelo Digital de Terreno da Sara Brasil (1930) — produto final,
│       resolução 10 m. Raster gerado por interpolação RST no GRASS GIS.
│
├── Sara_brasil_1930_modelo_tridimencional.glb
│       Modelo tridimensional do relevo (1930) em formato GLB.
│       Visualização 3D em web, Blender, Sketchfab, Twinmotion.
│
├── 01_dados_originais/
│   │   Dados brutos utilizados como entrada. NÃO MODIFICAR.
│   │
│   ├── raster_mapa_1897_georeferenciado.tif
│   │       Planta Geral da Capital de São Paulo (1897) — georreferenciada.
│   │
│   └── raster_sara_1930_curvas_5m.tif
│           Carta Sara Brasil (1930) — raster escaneado das curvas de 5 m.
│
├── 02_codigos/
│   │   Scripts Python do projeto, organizados por etapa.
│   │
│   ├── 01_extracao_curvas/
│   │   │   Etapa 1: extração automática de curvas e cotas.
│   │   │
│   │   ├── pipeline.txt .................. Descrição do fluxo desta etapa
│   │   ├── 01_inferir_curvas_unet.py ..... Segmentação de curvas com U-Net
│   │   ├── 02_extrair_cotas_ocr.py ....... Extração de cotas com EasyOCR
│   │   └── modelo_unet/
│   │       ├── unet_curvas_nivel.pt ...... Modelo U-Net treinado
│   │       └── unet_curvas_nivel_ep010.pt  Checkpoint da época 10
│   │
│   ├── 02_processamento_curvas/
│   │   │   Etapa 2: limpeza e vetorização das curvas extraídas.
│   │   │
│   │   ├── pipeline.txt ..................... Descrição do fluxo desta etapa
│   │   ├── 03_limpar_raster_curvas.py ....... Limpeza por filtro de forma
│   │   ├── 04_esqueletizar_vetorizar.py ..... Esqueletização + vetorização
│   │   ├── 05_conectar_gaps.py .............. Conexão de gaps entre pontas
│   │   ├── 06_snap_merge.py ................. Snap + merge em tiles
│   │   ├── 07_remover_galhos.py ............. Remoção de esporões
│   │   └── 08_limpar_topologia_grass.py ..... Limpeza topológica (GRASS)
│   │
│   └── 03_visualizacao_3d/
│       │   Etapa 3: geração de modelos tridimensionais.
│       │
│       └── 10_gerar_4_modelos.py ............ Gera 4 modelos OBJ do MDT
│
├── 03_desenvolvimento/
│   │   Arquivos de desenvolvimento, testes e intermediários.
│   │   Não compõem os produtos finais, mas documentam o processo.
│   │
│   ├── treinamento_unet/
│   │   ├── tiles_input_mascara/ ............. Tiles de treino (input + máscara)
│   │   ├── resultados_analise/ .............. Análise visual do modelo
│   │   └── resultados_teste_cego/ ........... Teste cego de validação
│   │
│   ├── testes_extracao_rgbhsv/ .............. Testes de extração por cor
│   │
│   ├── rasters_intermediarios/ .............. Rasters de etapas intermediárias
│   │   ├── curvas_5m_e_1m/
│   │   └── curvas_5m/
│   │
│   └── vetores_intermediarios/ .............. Vetores de etapas intermediárias
│       ├── etapa_1.gpkg
│       ├── etapa_2.gpkg
│       ├── etapa_3.gpkg
│       ├── etapa_4.gpkg
│       ├── etapa_5.gpkg
│       └── curvas_5m_e_1m/
│
├── 04_camadas_finais/
│   │   Produtos finais georreferenciados: vetores e rasters.
│   │
│   ├── vetoriais/
│   │   ├── sara_brasil_curvas_com_cota.gpkg ... Curvas vetorizadas com cota
│   │   ├── sara_brasil_curvas_sem_cota.gpkg ... Curvas sem cota atribuída
│   │   ├── sara_brasil_cotas.gpkg ............. Pontos de cota isolados
│   │   └── sara_brasil_cotas_rios.gpkg ........ Cotas extraídas de rios
│   │
│   └── rasters/
│       ├── sara_brasil_1930_raster_curvas.tif ..... Raster de curvas
│       ├── mdt_sara_brasil_1930_10m.tif ............ (cópia do raiz)
│       ├── hillshade_sara_brasil_1930.tif .......... Sombreamento do relevo
│       └── mapa_1897_com_relevo_1930.tif ........... Planta 1897 + relevo 1930
│
├── 05_mapas_publicacao/
│   │   Figuras finais.
│   │
│   ├── 01_curvas_raster.png .................. Curvas em raster
│   ├── 02_curvas_vetorizadas.png ............. Curvas vetorizadas
│   ├── 03_declividade.png .................... Mapa de declividade
│   ├── 04_hillshade.png ...................... Hillshade
│   ├── 05_hillshade_mapa_1897.png ............ Hillshade sobre planta 1897
│   ├── 06_mapa_altimetrico.png ............... Mapa altimétrico
│   └── 07_mapa_altimetrico_hipsometrico.png .. Mapa hipsométrico
│
└── 06_modelos_3d/
    │   Modelos tridimensionais gerados a partir do MDT.
    │   Centrados na origem (0,0,0) — para uso em Rhinoceros.
    │
    ├── modelo_1_geometria.obj ................. Geometria pura (escala real)
    ├── modelo_2_texturizado.obj ............... Geometria + textura
    ├── modelo_2_texturizado.mtl ............... Material do modelo 2
    ├── modelo_3_exagerado.obj ................. Geometria + exagero Z × 1.5
    ├── modelo_4_exagerado_texturizado.obj ..... Geometria + textura + exagero
    ├── modelo_4_exagerado_texturizado.mtl ..... Material do modelo 4
    ├── textura.png ............................ Textura aplicada (4096×4096)
    ├── Sara_brasil_1930_modelo_tridimencional.glb  (cópia do raiz)
    └── README.md .............................. Documentação dos modelos

--------------------------------------------------------------------------------
 NOTAS
--------------------------------------------------------------------------------

- Arquivos de sistema (.aux.xml, .qmd, etc.) podem existir em cada pasta.
  São gerados automaticamente pelo QGIS e podem ser ignorados.

- Todos os vetores estão em formato GeoPackage (.gpkg) para melhor
  compatibilidade e integridade.

- Todos os rasters estão em GeoTIFF (.tif) com compressão LZW.

- As figuras finais (05_mapas_publicacao/) estão em PNG, 300 dpi.

- Os modelos 3D estão em formato OBJ + MTL + PNG, prontos para
  importação em Rhinoceros, Blender, MeshLab ou similares.

- O arquivo GLB é uma versão alternativa em formato binário, adequada
  para visualização em web (Three.js, Sketchfab) e em motores 3D.

--------------------------------------------------------------------------------
 PIPELINE DE PROCESSAMENTO
--------------------------------------------------------------------------------

  01_dados_originais/
       │
       ▼
  [01_inferir_curvas_unet.py]  ← segmenta curvas da carta escaneada
       │
       ▼
  [03_limpar_raster_curvas.py]  ← remove ruído por filtro de forma
       │
       ▼
  [04_esqueletizar_vetorizar.py]  ← esqueletiza + vetoriza
       │
       ▼
  [05_conectar_gaps.py]  ← conecta pontas soltas
       │
       ▼
  [06_snap_merge.py]  ← snap + merge das extremidades
       │
       ▼
  [07_remover_galhos.py]  ← remove esporões finais
       │
       ▼
  04_camadas_finais/vetoriais/
       │
       ▼
  [Interpolação RST no GRASS GIS]
       │
       ▼
  MDT_SARA_BRASIL_1930_10m.tif  ← PRODUTO FINAL (na raiz)
       │
       ├──► [Hillshade] ──► 04_camadas_finais/rasters/hillshade_*.tif
       │
       ├──► [Morfometria] ──► declividade, TPI, aspect
       │
       ├──► [10_gerar_4_modelos.py] ──► 06_modelos_3d/*.obj
       │
       └──► [Export GLB] ──► Sara_brasil_1930_modelo_tridimencional.glb

  02_extrair_cotas_ocr.py  ← paralelo: extrai cotas via OCR
       │
       ▼
  04_camadas_finais/vetoriais/sara_brasil_cotas.gpkg

--------------------------------------------------------------------------------
 LICENÇA DE USO
--------------------------------------------------------------------------------

O conteúdo deste projeto é de livre acesso para fins acadêmicos, de pesquisa
e educacionais, desde que citado o autor. Qualquer uso comercial ou
redistribuição deve ser previamente autorizado.

Forma sugerida de citação:

    CHUERY PINTO, Matheus. Reconstrução do Relevo Paulistano:
    extração automática de curvas de nível do mapeamento Sara Brasil (1930)
    e geração de Modelos Digitais de Terreno. 2025–2027. Dissertação
    (Mestrado em Arquitetura e Urbanismo) – Faculdade de Arquitetura e
    Urbanismo e de Design, Universidade de São Paulo, São Paulo, 2027.

--------------------------------------------------------------------------------
 AGRADECIMENTOS
--------------------------------------------------------------------------------

O presente trabalho foi realizado com apoio do Conselho Nacional de
Desenvolvimento Científico e Tecnológico (CNPq – Brasil) – Processo
nº 133553/2025-6.


================================================================================

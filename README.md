# RECONSTRUÇÃO DO RELEVO PAULISTANO EXTRAÇÃO E MODELAGEM TOPOGRÁFICA DO SARA BRASIL (1930)

O projeto busca reconstruir o relevo histórico de São Paulo a partir do levantamento cartográfico Sara Brasil, de 1930, por meio de técnicas de geoprocessamento, aprendizado de máquina e automação computacional. O trabalho envolve a extração e vetorização de curvas de nível, o reconhecimento automatizado de cotas altimétricas e a geração de um Modelo Digital do Terreno (MDT). A base produzida constitui uma referência para investigar as transformações da topografia paulistana ao longo do tempo, a partir da comparação entre o relevo histórico e o atual.Todo o código-fonte utilizado neste projeto está sendo disponibilizado publicamente no GitHub.

Pesquisa de Mestrado
Faculdade de Arquitetura e Urbanismo e de Design
Universidade de São Paulo (FAU-USP)

Autor: Matheus Chuery Pinto
2026
Contato: matheuschuery@usp.br

O presente trabalho foi realizado com apoio do Conselho Nacional de
Desenvolvimento Científico e Tecnológico (CNPq – Brasil) – Processo
nº 133553/2025-6.

Todo o conteúdo deste projeto é de livre acesso, desde que citado o autor. Os arquivos completos estão disponíveis em https://doi.org/10.6084/m9.figshare.34625889 sob licença CC BY 4.0, que permite a reutilização e a adaptação dos materiais, desde que respeitadas as condições de atribuição.

The project aims to reconstruct the historical terrain of São Paulo using the 1930 Sara Brasil cartographic survey through geoprocessing, machine learning, and computational automation techniques. The workflow involves the extraction and vectorization of contour lines, the automated recognition of elevation values, and the generation of a Digital Terrain Model (DTM). The resulting dataset provides a foundation for investigating changes in São Paulo’s topography over time by comparing historical and present-day terrain models. All source code used in this project is being made publicly available on GitHub.

--------------------------------------------------------------------------------
 ARQUIVOS RAIZ
--------------------------------------------------------------------------------

# ESTRUTURA DO PROJETO — RECONSTRUÇÃO DO RELEVO PAULISTANO

## 1. ARQUIVOS PRINCIPAIS

1.1. `MDT_SARA_BRASIL_1930_10m.tif`

Modelo Digital do Terreno da Sara Brasil (1930), com resolução espacial de 10 m. Raster gerado por interpolação RST no GRASS GIS.

1.2. `Sara_brasil_1930_modelo_tridimencional.glb`

Modelo tridimensional do relevo histórico de 1930, em formato GLB, destinado à visualização em navegadores web e softwares de modelagem 3D.

1.3. `00_ESTRUTURA_PROJETO_RELEVO_PAULISTANO.txt`

Documento com a descrição da organização dos arquivos do projeto.

## 2. DADOS ORIGINAIS — `01_dados_originais/`

2.1. `raster_mapa_1897_georeferenciado.tif`

Planta Geral da Capital de São Paulo, de 1897, georreferenciada.

2.2. `raster_sara_1930_curvas_5m.tif`

Raster da carta Sara Brasil (1930), contendo as curvas de nível com equidistância de 5 m.

## 3. CÓDIGOS — `02_codigos/`

### 3.1. EXTRAÇÃO DE CURVAS — `01_extracao_curvas/`

3.1.1. `pipeline.txt` — Descrição do fluxo de processamento desta etapa.

3.1.2. `01_inferir_curvas_unet.py` — Segmentação automática das curvas de nível com U-Net.

3.1.3. `02_extrair_cotas_ocr.py` — Extração de cotas altimétricas com EasyOCR.

3.1.4. `modelo_unet/` — Diretório com os arquivos do modelo treinado.

3.1.5. `unet_curvas_nivel.pt` — Pesos do modelo U-Net.

3.1.6. `unet_curvas_nivel_ep010.pt` — Checkpoint da época 10.

### 3.2. PROCESSAMENTO DE CURVAS — `02_processamento_curvas/`

3.2.1. `pipeline.txt` — Descrição do fluxo de processamento desta etapa.

3.2.2. `03_limpar_raster_curvas.py` — Limpeza do raster por filtros de forma.

3.2.3. `04_esqueletizar_vetorizar.py` — Esqueletização e vetorização das curvas.

3.2.4. `05_conectar_gaps.py` — Conexão de segmentos interrompidos.

3.2.5. `06_snap_merge.py` — Ajuste e união de geometrias.

3.2.6. `07_remover_galhos.py` — Remoção de ramificações indesejadas.

3.2.7. `08_limpar_topologia_grass.py` — Limpeza topológica no GRASS GIS.

### 3.3. VISUALIZAÇÃO TRIDIMENSIONAL — `03_visualizacao_3d/`

3.3.1. `10_gerar_4_modelos.py` — Geração de quatro modelos tridimensionais em formato OBJ.

## 4. ARQUIVOS DE DESENVOLVIMENTO — `03_desenvolvimento/`

4.1. `treinamento_unet/` — Recortes de treinamento, máscaras e resultados de análise do modelo.

4.2. `testes_extracao_rgbhsv/` — Testes de extração de curvas por filtragem cromática.

4.3. `rasters_intermediarios/` — Rasters produzidos nas etapas intermediárias do processamento.

4.3.1. `curvas_5m_e_1m/` — Rasters com curvas de diferentes equidistâncias.

4.3.2. `curvas_5m/` — Rasters com curvas de equidistância de 5 m.

4.4. `vetores_intermediarios/` — Vetores produzidos durante as etapas de processamento.

4.4.1. `etapa_1.gpkg` — Vetores da etapa 1.

4.4.2. `etapa_2.gpkg` — Vetores da etapa 2.

4.4.3. `etapa_3.gpkg` — Vetores da etapa 3.

4.4.4. `etapa_4.gpkg` — Vetores da etapa 4.

4.4.5. `etapa_5.gpkg` — Vetores da etapa 5.

4.4.6. `curvas_5m_e_1m/` — Vetores intermediários das curvas de diferentes equidistâncias.

## 5. CAMADAS FINAIS — `04_camadas_finais/`

### 5.1. CAMADAS VETORIAIS — `vetoriais/`

5.1.1. `sara_brasil_curvas_com_cota.gpkg` — Curvas vetorizadas com cotas atribuídas.

5.1.2. `sara_brasil_curvas_sem_cota.gpkg` — Curvas vetorizadas sem cotas atribuídas.

5.1.3. `sara_brasil_cotas.gpkg` — Pontos altimétricos isolados.

5.1.4. `sara_brasil_cotas_rios.gpkg` — Pontos altimétricos associados às áreas dos rios.

### 5.2. CAMADAS RASTER — `rasters/`

5.2.1. `sara_brasil_1930_raster_curvas.tif` — Raster das curvas de nível extraídas.

5.2.2. `mdt_sara_brasil_1930_10m.tif` — Cópia do Modelo Digital do Terreno principal.

5.2.3. `hillshade_sara_brasil_1930.tif` — Raster de sombreamento do relevo.

5.2.4. `mapa_1897_com_relevo_1930.tif` — Planta de 1897 combinada à representação do relevo de 1930.

## 6. MAPAS PARA PUBLICAÇÃO — `05_mapas_publicacao/`

6.1. `01_curvas_raster.png` — Curvas de nível em formato raster.

6.2. `02_curvas_vetorizadas.png` — Curvas de nível vetorizadas.

6.3. `03_declividade.png` — Mapa de declividade.

6.4. `04_hillshade.png` — Mapa de sombreamento do relevo.

6.5. `05_hillshade_mapa_1897.png` — Sombreamento do relevo sobre a planta histórica de 1897.

6.6. `06_mapa_altimetrico.png` — Mapa altimétrico.

6.7. `07_mapa_altimetrico_hipsometrico.png` — Mapa hipsométrico.

## 7. MODELOS TRIDIMENSIONAIS — `06_modelos_3d/`

7.1. `modelo_1_geometria.obj` — Geometria do terreno em escala real.

7.2. `modelo_2_texturizado.obj` — Geometria do terreno com textura.

7.3. `modelo_2_texturizado.mtl` — Arquivo de materiais do modelo 2.

7.4. `modelo_3_exagerado.obj` — Modelo com exagero vertical de 1,5 vez.

7.5. `modelo_4_exagerado_texturizado.obj` — Modelo com exagero vertical e textura.

7.6. `modelo_4_exagerado_texturizado.mtl` — Arquivo de materiais do modelo 4.

7.7. `textura.png` — Textura aplicada aos modelos.

7.8. `Sara_brasil_1930_modelo_tridimencional.glb` — Modelo tridimensional em formato GLB.

7.9. `README.md` — Documentação dos modelos tridimensionais.

## 8. PIPELINE DE PROCESSAMENTO

8.1. ENTRADA DE DADOS

Utilização das cartas históricas georreferenciadas do mapa de 1897 e do levantamento Sara Brasil de 1930.

8.2. EXTRAÇÃO AUTOMÁTICA

Identificação das curvas de nível com U-Net e reconhecimento das cotas altimétricas com EasyOCR.

8.3. LIMPEZA DO RASTER

Remoção de ruídos e artefatos presentes nas curvas extraídas.

8.4. ESQUELETIZAÇÃO E VETORIZAÇÃO

Conversão das curvas rasterizadas em geometrias vetoriais.

8.5. CORREÇÃO GEOMÉTRICA E TOPOLÓGICA

Conexão de segmentos interrompidos, ajuste das geometrias e remoção de ramificações indesejadas.

8.6. GERAÇÃO DO MDT

Interpolação das informações altimétricas pelo método RST no GRASS GIS, com resolução de 10 m.

8.7. PRODUTOS DERIVADOS

Geração de mapas de sombreamento, declividade e representações altimétricas, além de modelos tridimensionais.

## 9. OBSERVAÇÕES

9.1. Os arquivos vetoriais utilizam o formato GeoPackage (`.gpkg`).

9.2. Os rasters utilizam o formato GeoTIFF (`.tif`).

9.3. Os arquivos intermediários documentam as etapas de processamento e desenvolvimento.

9.4. Arquivos auxiliares do QGIS, como `.aux.xml` e `.qmd`, podem ser ignorados.

9.5. Os modelos tridimensionais podem ser utilizados em softwares como Rhinoceros, Blender e MeshLab, conforme o formato do arquivo.

## 10. LICENÇA DE USO E CITAÇÃO

O conteúdo deste projeto é disponibilizado para fins acadêmicos, de pesquisa e educacionais, desde que o autor seja devidamente citado. Consulte a licença do repositório para verificar as condições de redistribuição e uso comercial.

FORMA SUGERIDA DE CITAÇÃO 

ABNT:

PINTO, Matheus Chuery. Reconstrução do relevo paulistano: extração e modelagem topográfica do Sara Brasil (1930). São Paulo, 2026. Dataset. DOI: https://doi.org/10.6084/m9.figshare.34625889.

APA:

Pinto, M. C. (2026). Reconstrução do relevo paulistano: Extração e modelagem topográfica do Sara Brasil (1930) [Dataset]. Figshare. https://doi.org/10.6084/m9.figshare.34625889

IEEE:

[1] M. C. Pinto, “Reconstrução do relevo paulistano: extração e modelagem topográfica do Sara Brasil (1930),” Figshare, 2026. [Dataset]. https://doi.org/10.6084/m9.figshare.34625889.

## 11. AGRADECIMENTOS

O presente trabalho foi realizado com apoio do Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq – Brasil), Processo nº 133553/2025-6.

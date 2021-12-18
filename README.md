# Correlaciones generalizadas en dinámica biomolecular

Este repositorio contiene el código del trabajo de tesis de licenciatura titulado "Impacto del uso de diferentes estimadores dedensidad de probabilidad conjunta en laidentificación de caminos alostéricos en proteínas". En el se abordan distintas técnica para medir correlaciones no lineales en la dinámica molecular basandose en teoría de la información, k nearest neighbors y kernel density estimation.

Make sure to create a new conda environment with the requirements.yml file

`$ conda env create -n gcc-env -f requirements.yml`


Example: 

`$ python pearson.py ../data/trj_displacement.npy`


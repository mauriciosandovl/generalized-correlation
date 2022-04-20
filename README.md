# Correlaciones generalizadas en dinámica biomolecular

Este repositorio contiene el código del trabajo de tesis de licenciatura titulado "Impacto del uso de diferentes estimadores de densidad de probabilidad conjunta en laidentificación de caminos alostéricos en proteínas". En el se abordan distintas técnica para medir correlaciones no lineales en la dinámica molecular basandose en teoría de la información, k nearest neighbors y kernel density estimation.

# Structure

```
generalized-correlation/
├── src/
│   ├──  kde/: Kernel Density Estimation
│   ├──  knn/: K Nearest Neighbors
│   ├──  lmi/: Linear Mutual Information
│   └──  pearson/: Pearson Correlation Coeficient
├── analysis/
|   └── Jupyter notebooks with analysis of the results
├── data/: Input files location
├── outputs/: Output files location
├── utils/: Auxiliar modules
└── requirements.txt: Dependencies list
```

# Install Dependencies

```$ pip install -r requirements.txt```

# Run Locally

```$ python src/<path>/<script_name>.py /data/<input_file_name>.npy```

Example `$ python src/pearson/pearson.py /data/trj_displacement.npy`

# More information
For more information contact: msandoval@ciencias.unam.mx or marciniega@ifc.unam.mx

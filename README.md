# Correlaciones generalizadas en dinámica biomolecular

Este repositorio contiene el código del trabajo de tesis de licenciatura titulado "Impacto del uso de diferentes estimadores dedensidad de probabilidad conjunta en laidentificación de caminos alostéricos en proteínas". En el se abordan distintas técnica para medir correlaciones no lineales en la dinámica molecular basandose en teoría de la información, k nearest neighbors y kernel density estimation.

# Structure

In the following diagram we include the most important directories and files in the repo with it's corresponding description

```
generalized-correlation/
├── models/
│   ├──  *.sql: all queries needed to create the collections file
│   └──  schema.yml: table and columns description with generic tests
├── macros/
|   └── jinja macros including additonal generic tests
├── dbt_project.yml: configuration file for dbt project
├── notify_update.py: python script to send google chat notification
├── requirements.txt: list of libraries and versions to be installed to run the
|       process
└── deploy.sh: bash script to execute commands `dbt run`, `dbt test` and send
        notification message
```

# Usage
## Prerequisites and setup
Make sure to create a new conda environment with the requirements.yml file

`$ conda env create -n gcc-env -f requirements.yml`

## Deploying
Example: 

`$ python pearson.py ../data/trj_displacement.npy`

# More information
For more information contact: msandoval@ciencias.unam.mx or marciniega@ifc.unam.mx


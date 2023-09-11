# Integración del algoritmo CSSVC y de un Wrapper de algoritmos de regresión en Orca-Python
Repositorio donde podremos encontrar el código fuente relacionado con el desarrollado de los algoritmos CSSVC y OrdinalWrapper, los cuales forman parte del proyecto de integración de algoritmos desde ORCA a ORCA-Python,
llevado a cabo por el grupo AYRNA de la Universidad de Córdoba.
Este proyecto se ha llevado acabo como Trabajo de Final de Grado del alumno Manuel Jesús Cabrera Delgado y tutorado por Pedro Antonio Gutiérrez Peña y David Guijo Rubio, miembros del grupo AYRNA.

## Clasificadores
### CSSVC
Parte de un modelo SVM de clasificación ordinal en el cual se hará uso de la descomposición 
OneVsAll. Los costes absolutos se incluyen como pesos diferentes para la clase negativa de cada descomposición. La clase negativa de cada uno de los problemas binarios generados a partir del método 
OneVsAll estará compuesta por todas las clases del problema original a excepción de la clase elegida como positiva en esa iteración.

### Ordinal Wrapper
Algoritmo implementado por primera vez para ORCA-Python orientado al uso de los regresores disponibles en la libreria scikit-learn, 
con el objetivo de permitir al usuario crear modelos ordinales y darle la capacidad de configurarlos como desee.

## Configuraciones
Ficheros en formato JSON a partir de los cuales se ejecutaran los experimentos dentro del framework ORCA-Python.
Los fichero estarán dividos en dos partes
### general-conf
Incluye la información básica para correr experimento, esta parte del fichero hace referencia a configuración especifica del experimento
- Localización local del los datasets
- Nombre de los datasets
- numero de "folds" empleados por la validación cruzada
- numero de "jobs" del GrindSearch
- tipo de preprocesamiento
- ruta de los ficheros de salida
- metricas

```
"general_conf": {
    "basedir": "/home/manuel/Escritorio/datasetDefs/non-discretized",
    "datasets": ["toy"],
    "input_preprocessing": "std",
    "hyperparam_cv_nfolds": 5,
    "jobs": 100,
    "output_folder": "my_runs/",
    "metrics": ["mae", "mze"],
    "cv_metric": "mae"
}
```
### configurations
Parte del archivo destinada a la configuración especifcada del álgoritmo empleado en el experimento.
```
"configurations": {
  "wrapper-SVR": {
    "classifier": "RegressorWrapper",
      "parameters": {
        "base_regressor": "sklearn.svm.SVR",
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "epsilon": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "gamma": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }
  }
}
```
### Test
Directorio donde podremos encontrar tests para validar el funcionamiento básico de los métodos, asi como, el tratamiento de errores.

### Datasets
Ejemplos de datasets empleados duramente la fase de experimento.

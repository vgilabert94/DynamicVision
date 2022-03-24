# OPTICAL FLOW
# Implementation Lukas-Kanade and Horn-Schunck methods.

## AUTOR

* **GILABERT MAÑO, VICENTE** - *Miembro 2* - [Vicent](https://github.com/vgilabert94)


## DESCRIPCION
Este repositorio contiene la práctica para la asignatura Aplicaciones Industriales en la Visión Artificial perteneciente al Máster Universitario en Visión Artificial impartido en la Universidad Rey Juan Carlos.  

El objetivo de este trabajo es la clasificación y detección de defectos en superficies metálicas en una línea de producción. Para ello realizaremos una implementación en *Python* con un detector de objetos (YOLOv5) y lo conectaremos mediante *C* a la aplicación que ya está funcionando en la fábrica.


## DOCUMENTACION
Se adjunta la lista de la documentación oficial del proyecto entregada al cliente:
* Entrega 1: -> [Especificación de Requisitos Software (ERS)](docs/ERS_grupoC.pdf)



## ESTRUCTURE DEL PROYECTO

```
.
├── dataset
│        ├── ANNOTATIONS
│        └── IMAGES
├── docs
│        └── ERS_grupoC.pdf
├── Exemples
│        ├── esquema.jpeg
│        ├── resultado.jpeg
│        ├── Screen1.png
│        └── tipos_defectos.png
├── pyproject.toml
├── README.md
├── requirements_dev.txt
├── requirements.txt
├── setup.cfg
├── setup.py
├── src
│   └── algorithm
│            ├── file_manager.py
│            ├── __init__.py
│            ├── main_algorithm.py
│            ├── py.typed
│            └── yolo_implementation.py
├── tests
│       └── test_algorithm.py
└── tox.ini
```


## RESULTADOS 
<p align="center">
	<img src="Exemples/resultado.jpeg" alt="resultado" width="50%"/>
</p>

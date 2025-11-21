# OmniStats Lab

"All-in-One" interactive lab built with Streamlit 🧪. Experiment with Inferential Statistics (Parametric/Non-Parametric), Machine Learning (Clustering, Regression), Deep Learning, and RL. Features synthetic data generation and real-time visualization with Plotly. Ideal for teaching and rapid prototyping. 📊🐍

[![Powered by PyMC-Marketing](https://img.shields.io/badge/Powered%20by-PyMC--Marketing-blue)](https://github.com/pymc-labs/pymc-marketing)

## Descripción

OmniStats Lab es una aplicación interactiva completa que permite experimentar con:
- **Estadística Inferencial**: Pruebas paramétricas y no paramétricas
- **Machine Learning**: Clustering, Regresión
- **Deep Learning**: Modelos de redes neuronales
- **Reinforcement Learning**: Algoritmos de aprendizaje por refuerzo
- **Marketing Science**: Marketing Mix Modeling (MMM) con inferencia bayesiana

La aplicación incluye generación de datos sintéticos y visualización en tiempo real con Plotly.

## Tecnologías

### Marketing Mix Modeling (MMM)
- **PyMC-Marketing** (Bayesian Inference v5): Framework moderno y mantenido por PyMC Labs para MMM
- **Migración completada**: El proyecto migró de `lightweight_mmm` (deprecado) a `pymc-marketing` para mayor modularidad y soporte activo
- **Características**: Modelado bayesiano, optimización de presupuesto, visualizaciones avanzadas

## Instalación

### Requisitos previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd omnistats-lab
```

2. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
```

3. Activa el entorno virtual:
   - **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   - **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. Instala las dependencias:
```bash
pip install -r requirements.txt
```

5. Ejecuta la aplicación:
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## Estructura del Proyecto

```
omnistats-lab/
├── app.py              # Aplicación principal Streamlit
├── requirements.txt    # Dependencias del proyecto
├── README.md          # Este archivo
├── .gitignore         # Archivos ignorados por Git
├── LICENSE            # Licencia del proyecto
└── assets/            # Recursos (imágenes, etc.)
```

## Uso

Una vez que la aplicación esté ejecutándose, podrás acceder a todas las funcionalidades a través de la interfaz web interactiva.

## Licencia

MIT License - Ver archivo LICENSE para más detalles.

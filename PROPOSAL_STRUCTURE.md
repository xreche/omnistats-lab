# Propuesta de Reestructuración: OmniStats Lab → Growth & Marketing Science Lab

## 📋 Análisis de la Estructura Actual

**Estado actual:**
- Aplicación Streamlit monolítica (`app.py`)
- Datos en `data/` sin organización por tipo
- Sin separación de lógica de negocio
- Sin estructura modular

**Objetivo:** Transformar en un repositorio profesional que demuestre competencias de Data Science para Growth/Marketing en D2C.

---

## 🏗️ Estructura Propuesta (Cookiecutter Data Science)

```
omnistats-lab/
│
├── README.md                          # Documentación principal actualizada
├── LICENSE
├── requirements.txt                   # Dependencias actualizadas
├── .gitignore
├── .env.example                       # Variables de entorno (API keys)
│
├── data/                              # Datos organizados por tipo
│   ├── raw/                          # Datos originales (no modificar)
│   │   ├── brazilian-ecommerce/      # Dataset Olist completo
│   │   └── marketing-mix/           # Dataset MMM sintético
│   ├── processed/                    # Datos transformados
│   │   ├── customer_analytics/       # Datos para análisis de clientes
│   │   ├── mmm/                      # Datos para Marketing Mix Modeling
│   │   └── causal/                   # Datos para inferencia causal
│   └── external/                     # Datos externos (si aplica)
│
├── notebooks/                         # Jupyter notebooks exploratorios
│   ├── 01_customer_analytics/        # Exploración de KPIs de cliente
│   ├── 02_marketing_science/         # Exploración MMM y elasticidad
│   ├── 03_causal_inference/          # Análisis causal
│   └── 04_genai/                     # Experimentos con GenAI
│
├── src/                              # Código fuente modular
│   ├── __init__.py
│   │
│   ├── data/                         # Pipeline de datos
│   │   ├── __init__.py
│   │   ├── loaders.py                # Funciones para cargar datasets
│   │   ├── cleaners.py               # Limpieza y transformación
│   │   └── validators.py             # Validación de datos
│   │
│   ├── features/                     # Feature engineering
│   │   ├── __init__.py
│   │   ├── customer_features.py     # Features de cliente (RFM, etc.)
│   │   ├── marketing_features.py    # Features de marketing (Adstock, etc.)
│   │   └── temporal_features.py     # Features temporales
│   │
│   ├── models/                       # Modelos organizados por pilar
│   │   ├── __init__.py
│   │   │
│   │   ├── customer_analytics/      # PILAR 1
│   │   │   ├── __init__.py
│   │   │   ├── cac.py               # Cálculo de CAC
│   │   │   ├── ltv.py               # Lifetime Value (BG/NBD, Gamma-Gamma)
│   │   │   ├── churn.py             # Churn rate
│   │   │   └── rfm.py               # Segmentación RFM
│   │   │
│   │   ├── marketing_science/       # PILAR 2
│   │   │   ├── __init__.py
│   │   │   ├── mmm.py               # Marketing Mix Modeling
│   │   │   ├── price_elasticity.py  # Elasticidad de precio
│   │   │   └── attribution.py       # Atribución Multi-Touch
│   │   │
│   │   ├── causal_inference/        # PILAR 3
│   │   │   ├── __init__.py
│   │   │   ├── psm.py               # Propensity Score Matching
│   │   │   └── did.py               # Difference-in-Differences
│   │   │
│   │   └── genai/                   # PILAR 4
│   │       ├── __init__.py
│   │       ├── rag.py               # RAG pipeline
│   │       └── content_generation.py # Generación de contenido
│   │
│   ├── utils/                        # Utilidades compartidas
│   │   ├── __init__.py
│   │   ├── logging_config.py        # Configuración de logging
│   │   ├── exceptions.py             # Excepciones personalizadas
│   │   └── helpers.py               # Funciones auxiliares
│   │
│   └── visualization/               # Visualizaciones
│       ├── __init__.py
│       ├── kpi_dashboards.py        # Dashboards de KPIs
│       └── model_plots.py            # Visualizaciones de modelos
│
├── config/                           # Configuraciones
│   ├── config.yaml                   # Configuración principal
│   ├── model_configs/                # Configuraciones por modelo
│   │   ├── mmm_config.yaml
│   │   └── ltv_config.yaml
│   └── logging.yaml                  # Configuración de logging
│
├── tests/                            # Tests unitarios e integración
│   ├── __init__.py
│   ├── test_customer_analytics/
│   ├── test_marketing_science/
│   ├── test_causal_inference/
│   └── test_genai/
│
├── scripts/                          # Scripts ejecutables
│   ├── run_customer_analytics.py    # Pipeline completo de análisis de clientes
│   ├── run_mmm.py                    # Pipeline MMM
│   ├── run_causal_analysis.py       # Análisis causal
│   └── run_genai_pipeline.py        # Pipeline GenAI
│
├── app/                              # Aplicación Streamlit (refactorizada)
│   ├── __init__.py
│   ├── main.py                       # App principal
│   ├── pages/                        # Páginas por pilar
│   │   ├── 1_customer_analytics.py
│   │   ├── 2_marketing_science.py
│   │   ├── 3_causal_inference.py
│   │   └── 4_genai.py
│   └── components/                   # Componentes reutilizables
│       ├── kpi_cards.py
│       └── charts.py
│
├── docs/                             # Documentación
│   ├── architecture.md               # Arquitectura del sistema
│   ├── api_reference.md              # Referencia de API
│   └── methodology/                  # Metodologías por pilar
│       ├── customer_analytics.md
│       ├── marketing_science.md
│       ├── causal_inference.md
│       └── genai.md
│
└── outputs/                          # Resultados y artefactos
    ├── models/                       # Modelos entrenados (pickle/joblib)
    ├── reports/                      # Reportes generados
    ├── visualizations/               # Gráficos guardados
    └── predictions/                 # Predicciones

```

---

## 🎯 Mapeo de Pilares a Estructura

### PILAR 1: Customer Analytics & KPIs
**Ubicación:** `src/models/customer_analytics/`
- `cac.py` → CAC Blended y por Canal
- `ltv.py` → BG/NBD y Gamma-Gamma (lifetimes)
- `churn.py` → Churn rate (ventana 90 días)
- `rfm.py` → Segmentación RFM

**Scripts:** `scripts/run_customer_analytics.py`
**Notebooks:** `notebooks/01_customer_analytics/`
**App:** `app/pages/1_customer_analytics.py`

---

### PILAR 2: Econometría y Marketing Science
**Ubicación:** `src/models/marketing_science/`
- `mmm.py` → lightweight_mmm (Adstock + Hill)
- `price_elasticity.py` → Regresión Log-Log (OLS)
- `attribution.py` → Cadenas de Markov

**Scripts:** `scripts/run_mmm.py`
**Notebooks:** `notebooks/02_marketing_science/`
**App:** `app/pages/2_marketing_science.py`

---

### PILAR 3: Inferencia Causal
**Ubicación:** `src/models/causal_inference/`
- `psm.py` → DoWhy (Propensity Score Matching)
- `did.py` → Difference-in-Differences

**Scripts:** `scripts/run_causal_analysis.py`
**Notebooks:** `notebooks/03_causal_inference/`
**App:** `app/pages/3_causal_inference.py`

---

### PILAR 4: GenAI & Automatización
**Ubicación:** `src/models/genai/`
- `rag.py` → LangChain + ChromaDB
- `content_generation.py` → OpenAI API (emails personalizados)

**Scripts:** `scripts/run_genai_pipeline.py`
**Notebooks:** `notebooks/04_genai/`
**App:** `app/pages/4_genai.py`

---

## 📦 Dependencias Propuestas (requirements.txt)

```txt
# Core Data Science
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Customer Analytics
lifetimes>=0.11.3          # BG/NBD, Gamma-Gamma

# Marketing Science
lightweight-mmm>=0.1.0     # Google MMM
statsmodels>=0.14.0        # Econometría
pymc>=5.0.0                # Bayesian modeling (opcional)

# Causal Inference
dowhy>=0.11.0              # Causal inference
econml>=0.14.0             # Causal ML (opcional)

# GenAI
langchain>=0.1.0
chromadb>=0.4.0
openai>=1.0.0

# Visualization
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0

# App
streamlit>=1.28.0

# Utilities
pyyaml>=6.0                 # Config files
python-dotenv>=1.0.0       # Environment variables
tqdm>=4.65.0               # Progress bars
joblib>=1.3.0              # Model serialization

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Code Quality
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
```

---

## 🔧 Estándares de Código

1. **Type Hints:** Todos los parámetros y retornos tipados
2. **Docstrings:** Google-style para todas las funciones/clases
3. **Error Handling:** Try/except con logging apropiado
4. **Validación:** Validación de inputs con mensajes claros
5. **Logging:** Uso de logging module (no prints)
6. **Testing:** Tests unitarios para funciones críticas

---

## 📝 Próximos Pasos

1. **Confirmar estructura** (este documento)
2. **Generar código del PILAR 1** (Customer Analytics)
3. **Actualizar requirements.txt**
4. **Crear configuraciones base**
5. **Migrar app.py a estructura modular**

---

## ❓ Preguntas para Confirmación

1. ¿La estructura propuesta cubre tus expectativas?
2. ¿Algún ajuste en la organización de carpetas?
3. ¿Priorizamos algún pilar específico?
4. ¿Incluimos tests desde el inicio o después?

**Esperando tu confirmación para comenzar con el PILAR 1: Customer Analytics & KPIs** 🚀


# Configuración de Marketing Mix Modeling (MMM)

Este documento explica cómo configurar y usar el Marketing Mix Modeling en OmniStats Lab usando **PyMC-Marketing**.

## Tecnología: PyMC-Marketing

El proyecto utiliza **PyMC-Marketing**, el estándar moderno para MMM mantenido por PyMC Labs. Esta migración desde `lightweight_mmm` (deprecado) proporciona:

- ✅ **Soporte activo**: Mantenido por PyMC Labs
- ✅ **Modelado bayesiano completo**: Inferencia con PyMC v5
- ✅ **Optimización de presupuesto**: Funcionalidad integrada
- ✅ **Visualizaciones avanzadas**: Integración con ArviZ
- ✅ **Mayor flexibilidad**: Control sobre priors y estructura del modelo

## Instalación

### Requisitos

Las dependencias se instalan automáticamente con:

```bash
pip install -r requirements.txt
```

Esto incluye:
- `pymc-marketing>=0.5.0` - Framework principal para MMM
- `pymc>=5.0.0` - Framework de programación probabilística
- `arviz>=0.17.0` - Análisis y visualización bayesiana
- `xarray>=2023.0.0` - Arrays multidimensionales

### Verificación

Para verificar que PyMC-Marketing está instalado correctamente:

```python
from pymc_marketing.mmm import DelayedSaturatedMMM
print("PyMC-Marketing instalado correctamente")
```

## Uso Básico

### Ejecutar el Pipeline Completo

```bash
python scripts/run_marketing_science.py
```

El script:
1. Carga los datos de Marketing Mix
2. Entrena el modelo MMM con PyMC-Marketing
3. Genera visualizaciones automáticamente
4. Guarda resultados en `outputs/reports/mmm_results.txt`

### Uso Programático

```python
from src.models.marketing_science import run_mmm_analysis, optimize_channel_budget

# Entrenar modelo
mmm_results = run_mmm_analysis(
    df=data,
    target_col='sales',
    media_channels=['tv_spend', 'radio_spend', 'digital_spend'],
    control_vars=['price'],
    date_col='date',
    adstock_max_lag=8,
    yearly_seasonality=2,
    draws=1000,
    chains=2,
    tune=1000
)

# Optimizar presupuesto
budget_allocation = optimize_channel_budget(
    mmm_model=mmm_results['model'],
    total_budget=1000000,
    budget_bounds={
        'tv_spend': (0, 500000),
        'radio_spend': (0, 300000),
        'digital_spend': (0, 200000)
    }
)
```

## Configuración del Modelo

### Parámetros Principales

- **`adstock_max_lag`** (default: 8): Lag máximo para la transformación Adstock
- **`yearly_seasonality`** (default: 2): Número de términos de Fourier para estacionalidad anual
- **`draws`** (default: 1000): Número de muestras posteriores
- **`chains`** (default: 2): Número de cadenas MCMC
- **`tune`** (default: 1000): Número de muestras de calentamiento

### Requisitos de Datos

El modelo requiere:
- **Columna de fecha**: Debe estar en formato datetime
- **Canales de medios**: Columnas numéricas con gastos por canal
- **Variable objetivo**: Columna con las ventas o métrica objetivo
- **Variables de control** (opcional): Precio, promociones, etc.

Ejemplo de estructura de datos:

```python
data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=104, freq='W'),
    'sales': [1000, 1200, ...],
    'tv_spend': [500, 600, ...],
    'radio_spend': [200, 250, ...],
    'digital_spend': [300, 350, ...],
    'price': [10.0, 10.5, ...]
})
```

## Visualizaciones

Las visualizaciones se generan automáticamente y se guardan en `outputs/visualizations/mmm/`:

- **Component Contributions**: Contribución de cada canal al objetivo
- **Direct Contribution Curves**: Curvas de respuesta directa por canal

## Optimización de Presupuesto

La función `optimize_channel_budget()` encuentra la asignación óptima de presupuesto:

```python
from src.models.marketing_science import optimize_channel_budget

allocation = optimize_channel_budget(
    mmm_model=mmm_results['model'],
    total_budget=1000000,
    budget_bounds={
        'tv_spend': (0, 500000),      # Min y max por canal
        'radio_spend': (0, 300000),
        'digital_spend': (0, 200000)
    }
)

print(allocation['optimal_allocation'])
```

## Interpretación de Resultados

### Media Effectiveness

Los resultados incluyen estadísticas posteriores para cada canal:

- **Mean**: Media de la distribución posterior
- **Median**: Mediana de la distribución posterior
- **HDI (Highest Density Interval)**: Intervalo de credibilidad del 95%

### InferenceData

El modelo retorna un objeto `InferenceData` de ArviZ que permite:

- Diagnósticos MCMC (R-hat, ESS)
- Visualizaciones con ArviZ
- Análisis posterior avanzado

```python
import arviz as az

# Acceder a los datos de inferencia
idata = mmm_results['idata']

# Ver resumen
az.summary(idata)

# Generar diagnósticos
az.plot_trace(idata)
```

## Solución de Problemas

### Error: "pymc-marketing not available"

Instala las dependencias:
```bash
pip install pymc-marketing pymc arviz xarray
```

### Error: "date column not found"

Asegúrate de que tu DataFrame tenga una columna de fecha:
```python
data['date'] = pd.to_datetime(data['date'])
```

### Modelo tarda mucho en entrenar

Reduce el número de muestras:
```python
mmm_results = run_mmm_analysis(
    ...,
    draws=500,  # Reducir de 1000 a 500
    tune=500    # Reducir de 1000 a 500
)
```

## Referencias

- [PyMC-Marketing Documentation](https://www.pymc-labs.io/blog-posts/pymc-marketing/)
- [PyMC Documentation](https://www.pymc.io/)
- [ArviZ Documentation](https://www.arviz.org/)
- [Migración desde lightweight_mmm](MIGRATION_PYMC_MARKETING.md)

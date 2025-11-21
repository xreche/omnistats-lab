# Migración a PyMC-Marketing

Este documento describe la migración del módulo de Marketing Mix Modeling (MMM) de `lightweight_mmm` (Google) a `pymc-marketing` (PyMC Labs).

## Razón de la Migración

- **lightweight_mmm está deprecado**: Google ya no mantiene activamente esta librería
- **Red flag para portfolio**: El uso de librerías deprecadas es una señal negativa para reclutadores
- **PyMC-Marketing es el estándar moderno**: Mantenido activamente por PyMC Labs
- **Mayor modularidad**: PyMC-Marketing ofrece más flexibilidad y control
- **Soporte activo**: Comunidad activa y actualizaciones regulares

## Cambios Realizados

### 1. Dependencias (`requirements.txt`)

**Eliminado:**
- `lightweight-mmm>=0.1.0`

**Añadido:**
- `pymc-marketing>=0.5.0` - Framework principal para MMM
- `pymc>=5.0.0` - Framework de programación probabilística
- `arviz>=0.17.0` - Análisis y visualización bayesiana
- `xarray>=2023.0.0` - Arrays multidimensionales para outputs de inferencia

### 2. Refactorización del Código (`src/models/marketing_science/mmm.py`)

#### Antes (lightweight_mmm):
```python
from lightweight_mmm import lightweight_mmm

mmm = lightweight_mmm.LightweightMMM()
mmm.fit(
    media=media,
    media_prior=media_prior,
    target=target,
    extra_features=control,
    number_samples=n_samples,
    number_chains=n_chains,
    number_warmup=n_warmup
)
```

#### Después (pymc-marketing):
```python
from pymc_marketing.mmm import DelayedSaturatedMMM

mmm = DelayedSaturatedMMM(
    date_column="date",
    channel_columns=["tv", "radio", "digital"],
    adstock_max_lag=8,
    yearly_seasonality=2,
)

mmm.fit(
    X=data[channel_cols],
    y=data[target_col],
    target_column="sales",
    draws=1000,
    chains=2,
    tune=1000
)
```

### 3. Nuevas Funcionalidades

#### Optimización de Presupuesto
```python
from src.models.marketing_science import optimize_channel_budget

budget_allocation = optimize_channel_budget(
    mmm_model=mmm_results['model'],
    total_budget=1000000,
    budget_bounds={
        'tv': (0, 500000),
        'radio': (0, 300000),
        'digital': (0, 200000)
    }
)
```

#### Visualizaciones
```python
from src.models.marketing_science import plot_mmm_results

plot_mmm_results(
    mmm_model=mmm_results['model'],
    output_path="outputs/visualizations/mmm"
)
```

### 4. Cambios en la API

| Aspecto | lightweight_mmm | pymc-marketing |
|---------|-----------------|----------------|
| **Entrada de datos** | Arrays NumPy | DataFrames de Pandas |
| **Columna de fecha** | Opcional | Requerida explícitamente |
| **Transformaciones** | Manuales (Adstock/Saturación) | Integradas en el modelo |
| **Parámetros MCMC** | `number_samples`, `number_warmup` | `draws`, `tune` |
| **Outputs** | `trace` dict | `InferenceData` (ArviZ) |
| **Optimización** | No incluida | `optimize_channel_budget()` |

## Uso Actualizado

### Ejecutar el Pipeline

```bash
python scripts/run_marketing_science.py
```

El script ahora:
1. Usa `pymc-marketing` para el modelado bayesiano
2. Genera visualizaciones automáticamente
3. Guarda resultados en formato compatible con ArviZ

### Configuración del Modelo

Los parámetros principales del modelo se pueden ajustar en `run_marketing_science.py`:

```python
mmm_results = run_mmm_analysis(
    df=mmm_data,
    target_col='sales',
    media_channels=['tv_spend', 'radio_spend', 'digital_spend'],
    control_vars=['price'],
    date_col='date',
    adstock_max_lag=8,        # Lag máximo para adstock
    yearly_seasonality=2,     # Términos de Fourier para estacionalidad
    draws=1000,               # Muestras posteriores
    chains=2,                 # Cadenas MCMC
    tune=1000,                # Muestras de calentamiento
    random_seed=42            # Semilla para reproducibilidad
)
```

## Ventajas de la Migración

1. **Soporte activo**: PyMC Labs mantiene y actualiza regularmente la librería
2. **Mejor integración**: Compatible con el ecosistema PyMC/ArviZ
3. **Más flexible**: Mayor control sobre priors y estructura del modelo
4. **Visualizaciones**: Integración nativa con ArviZ para diagnósticos
5. **Optimización**: Funcionalidad de optimización de presupuesto incluida
6. **Portfolio**: Demuestra conocimiento de tecnologías modernas y mantenidas

## Notas de Compatibilidad

- Los datos deben tener una columna de fecha explícita
- Los canales de medios deben estar en formato DataFrame (no arrays)
- Las transformaciones Adstock y Saturación están integradas en el modelo
- Los outputs son objetos `InferenceData` de ArviZ, más ricos que los traces anteriores

## Referencias

- [PyMC-Marketing Documentation](https://www.pymc-labs.io/blog-posts/pymc-marketing/)
- [PyMC Documentation](https://www.pymc.io/)
- [ArviZ Documentation](https://www.arviz.org/)


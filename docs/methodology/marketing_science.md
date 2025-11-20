# Marketing Science - Metodología

## Descripción General

Este módulo implementa modelos econométricos avanzados para análisis de marketing:

1. **MMM (Marketing Mix Modeling)**: Modelado del impacto de medios en ventas
2. **Price Elasticity**: Elasticidad precio de la demanda
3. **Multi-Touch Attribution**: Atribución usando Cadenas de Markov

---

## 1. Marketing Mix Modeling (MMM)

### Definición

MMM es un modelo estadístico que cuantifica el impacto de diferentes canales de marketing en las ventas, considerando efectos de carryover (Adstock) y saturación.

### Transformaciones

#### Adstock (Carryover Effect)

Modela el efecto residual de la publicidad pasada:

```
adstock_t = spend_t + decay × adstock_{t-1}
```

- **decay**: Parámetro de decaimiento (0 < decay < 1)
- Mayor decay = efecto más largo

#### Saturation (Hill Function)

Modela rendimientos decrecientes:

```
saturated = (spend^slope) / (half_saturation^slope + spend^slope)
```

- **half_saturation**: Nivel de gasto al 50% del efecto máximo
- **slope**: Pendiente de la curva

### Implementación

**Ubicación:** `src/models/marketing_science/mmm.py`

**Librería:** `lightweight_mmm` (Google)

**Uso:**
```python
from src.models.marketing_science import run_mmm_analysis

results = run_mmm_analysis(
    df=data,
    target_col='sales',
    media_channels=['tv_spend', 'radio_spend', 'digital_spend'],
    control_vars=['price', 'promotion'],
    apply_transformations=True
)
```

---

## 2. Price Elasticity

### Definición

La elasticidad precio mide la sensibilidad de la demanda a cambios en el precio.

**Modelo Log-Log:**
```
log(quantity) = α + β × log(price) + γ × promotion + controls + ε
```

**Elasticidad = β** (coeficiente de log(price))

- **Elastic (β < -1)**: Demanda cambia más que precio
- **Inelastic (β > -1)**: Demanda cambia menos que precio

### Implementación

**Ubicación:** `src/models/marketing_science/price_elasticity.py`

**Librería:** `statsmodels` (OLS)

**Uso:**
```python
from src.models.marketing_science import calculate_price_elasticity

results = calculate_price_elasticity(
    df=data,
    quantity_col='quantity',
    price_col='price',
    promotion_col='promotion',
    control_vars=['competitor_price']
)
```

---

## 3. Multi-Touch Attribution (Markov Chains)

### Definición

Atribuye conversiones a múltiples touchpoints usando cadenas de Markov para modelar la probabilidad de conversión en diferentes caminos del cliente.

### Metodología

1. **Construir Cadenas de Markov**: Transiciones entre touchpoints
2. **Calcular Removal Effect**: Impacto de remover cada touchpoint
3. **Atribuir Conversiones**: Basado en removal effects

**Removal Effect:**
```
Removal Effect = (Baseline Conversion - Conversion without touchpoint) / Baseline Conversion
```

**Attribution Score:**
```
Attribution = Removal Effect × Total Conversions
```

### Implementación

**Ubicación:** `src/models/marketing_science/attribution.py`

**Uso:**
```python
from src.models.marketing_science import calculate_markov_attribution

results = calculate_markov_attribution(
    customer_journeys=journeys_df,
    touchpoint_col='touchpoint',
    conversion_col='conversion',
    customer_id_col='customer_id'
)
```

---

## Referencias

- **Lightweight MMM**: https://github.com/google/lightweight-mmm
- **Price Elasticity**: Wooldridge, J. M. (2015). Introductory Econometrics
- **Markov Attribution**: Anderl et al. (2016). Mapping Customer Touchpoints


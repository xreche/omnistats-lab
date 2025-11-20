# Customer Analytics & KPIs - Metodología

## Descripción General

Este módulo implementa las métricas fundamentales de análisis de clientes para empresas D2C (Direct-to-Consumer), incluyendo:

1. **CAC (Customer Acquisition Cost)**: Coste de adquisición de clientes
2. **LTV (Lifetime Value)**: Valor de vida del cliente
3. **Churn Rate**: Tasa de abandono
4. **RFM Segmentation**: Segmentación basada en Recencia, Frecuencia y Monetización

---

## 1. CAC (Customer Acquisition Cost)

### Definición

El CAC mide cuánto cuesta adquirir un nuevo cliente. Se calcula como:

```
CAC = Total Marketing Spend / Número de Nuevos Clientes
```

### Implementación

**Ubicación:** `src/models/customer_analytics/cac.py`

**Funciones principales:**
- `calculate_cac()`: Calcula CAC blended o por canal
- `calculate_cac_by_channel()`: Calcula CAC desglosado por canal de marketing

**Uso:**
```python
from src.models.customer_analytics import calculate_cac

# CAC Blended
cac = calculate_cac(
    marketing_spend=spend_df,
    new_customers=customers_df,
    date_col='date',
    spend_col='spend',
    customer_id_col='customer_id'
)

# CAC por Canal
cac_by_channel = calculate_cac_by_channel(
    marketing_spend=spend_df,
    new_customers=customers_df,
    channel_col='channel'
)
```

---

## 2. LTV (Lifetime Value)

### Definición

El LTV predice el valor total que un cliente generará durante su relación con la empresa.

### Metodología

Utilizamos modelos probabilísticos de la librería `lifetimes`:

1. **BG/NBD (Beta Geometric / Negative Binomial Distribution)**
   - Predice la frecuencia futura de compras
   - Modela cuándo un cliente hará su próxima compra
   - Parámetros: α, r, a, b

2. **Gamma-Gamma**
   - Predice el valor promedio por transacción
   - Modela la heterogeneidad en el valor de las transacciones
   - Parámetros: p, q, v

**Cálculo de LTV:**
```
LTV = Predicted Transactions × Predicted Avg Order Value × Discount Factor
```

### Implementación

**Ubicación:** `src/models/customer_analytics/ltv.py`

**Función principal:**
- `calculate_ltv()`: Calcula LTV usando BG/NBD y Gamma-Gamma

**Parámetros clave:**
- `prediction_period_days`: Período de predicción (default: 365 días)
- `discount_rate`: Tasa de descuento para flujos futuros (default: 0.1 = 10%)

**Uso:**
```python
from src.models.customer_analytics import calculate_ltv

ltv_df, metrics = calculate_ltv(
    orders_df=orders,
    customer_id_col='customer_id',
    date_col='order_purchase_timestamp',
    value_col='order_value',
    prediction_period_days=365,
    discount_rate=0.1
)
```

---

## 3. Churn Rate

### Definición

Para empresas no-suscripción, el churn se define como clientes que no han realizado una compra dentro de una ventana de inactividad (default: 90 días).

### Metodología

```
Churn Rate = (Clientes Churned / Total Clientes) × 100
```

Un cliente está "churned" si:
- `days_since_last_purchase > inactivity_window_days`

### Implementación

**Ubicación:** `src/models/customer_analytics/churn.py`

**Función principal:**
- `calculate_churn_rate()`: Calcula tasa de churn

**Uso:**
```python
from src.models.customer_analytics import calculate_churn_rate

churn_df, metrics = calculate_churn_rate(
    orders_df=orders,
    customer_id_col='customer_id',
    date_col='order_purchase_timestamp',
    inactivity_window_days=90
)
```

---

## 4. RFM Segmentation

### Definición

RFM segmenta clientes basándose en:

- **Recency (R)**: Días desde la última compra
- **Frequency (F)**: Número de compras totales
- **Monetary (M)**: Valor total gastado

Cada dimensión se puntúa de 1 a 5 (5 = mejor).

### Metodología

1. **Cálculo de Scores:**
   - Recency: Quintiles inversos (más reciente = score más alto)
   - Frequency: Quintiles (más compras = score más alto)
   - Monetary: Quintiles (más gasto = score más alto)

2. **Segmentación:**
   - Combina scores R-F-M (ej: "555" = Champions)
   - Asigna segmentos predefinidos (Champions, Loyal Customers, etc.)

### Segmentos Estándar

- **Champions** (555, 554, etc.): Mejores clientes
- **Loyal Customers**: Clientes frecuentes
- **Potential Loyalists**: Clientes prometedores
- **New Customers**: Clientes nuevos
- **At Risk**: Clientes en riesgo de churn
- **Cannot Lose Them**: Clientes de alto valor
- **About to Sleep**: Clientes inactivos

### Implementación

**Ubicación:** `src/models/customer_analytics/rfm.py`

**Funciones principales:**
- `calculate_rfm_scores()`: Calcula scores RFM
- `assign_rfm_segments()`: Asigna segmentos
- `calculate_rfm_segments()`: Pipeline completo

**Uso:**
```python
from src.models.customer_analytics import calculate_rfm_segments

rfm_df, metrics = calculate_rfm_segments(
    orders_df=orders,
    customer_id_col='customer_id',
    date_col='order_purchase_timestamp',
    value_col='order_value'
)
```

---

## Pipeline Completo

**Script ejecutable:** `scripts/run_customer_analytics.py`

Ejecuta todos los análisis en secuencia:

```bash
python scripts/run_customer_analytics.py
```

**Outputs:**
- `outputs/reports/rfm_segments.csv`
- `outputs/reports/ltv_predictions.csv`
- `outputs/reports/churn_analysis.csv`

---

## Referencias

- **Lifetimes Library**: https://github.com/CamDavidsonPilon/lifetimes
- **RFM Segmentation**: https://www.putler.com/rfm-analysis/
- **BG/NBD Model**: Fader, Hardie, Lee (2005)


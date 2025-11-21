# Tareas Pendientes - OmniStats Lab

## ✅ PILAR 2: Marketing Science - COMPLETADO

### Estado Actual
- ✅ **Migración completada:** `lightweight-mmm` → `pymc-marketing`
- ✅ **Dependencias instaladas:** `pymc-marketing>=0.5.0`, `pymc>=5.0.0`, `arviz>=0.17.0`, `xarray>=2023.0.0`
- ✅ **Código refactorizado:** `src/models/marketing_science/mmm.py` usando PyMC-Marketing
- ✅ **Pipeline funcional:** `scripts/run_marketing_science.py` ejecuta correctamente
- ✅ **Información de progreso:** Logging detallado durante el entrenamiento MCMC

### Mejoras Pendientes (No Críticas)

#### 1. Optimización de Convergencia MCMC
- [ ] Aumentar `target_accept` en `mmm.fit()` para reducir divergencias
  - Actualmente: 44 divergencias con parámetros mínimos
  - Objetivo: <5 divergencias con parámetros de producción
  - Ubicación: `src/models/marketing_science/mmm.py` línea ~219
  - Solución: Añadir `target_accept=0.95` o `0.99` al `mmm.fit()`

#### 2. Corrección de Visualizaciones
- [ ] Arreglar `plot_channel_contribution_grid()` - requiere argumentos `start`, `stop`, `num`
  - Ubicación: `src/models/marketing_science/mmm.py` línea ~405
  - Solución: Pasar parámetros temporales desde el DataFrame o configuración
  
- [ ] Arreglar `plot_allocated_contribution_by_channel()` - requiere argumento `samples`
  - Ubicación: `src/models/marketing_science/mmm.py` línea ~417
  - Solución: Extraer muestras del `idata` posterior

#### 3. Extracción de Efectividad de Medios
- [ ] Mejorar método `get_channel_contributions_posterior()` o `get_ts_contribution_posterior()`
  - Actualmente: Usa fallback method
  - Objetivo: Extraer contribuciones correctamente desde el posterior
  - Ubicación: `src/models/marketing_science/mmm.py` línea ~239

#### 4. Parámetros de Producción
- [ ] Documentar parámetros recomendados para producción:
  - `draws=1000` (actualmente 50 para pruebas)
  - `tune=1000` (actualmente 50 para pruebas)
  - `chains=2` (actualmente 1 para pruebas)
  - `target_accept=0.95` (nuevo parámetro a añadir)

---

## 📋 Pilares Pendientes de Implementación

### Pilar 3: Inferencia Causal - ✅ IMPLEMENTADO

#### Estado Actual
- ✅ **Propensity Score Matching (PSM)**: Implementado en `src/models/causal_inference/psm.py`
- ✅ **Difference-in-Differences (DiD)**: Implementado en `src/models/causal_inference/did.py`
- ✅ **Script de ejecución**: `scripts/run_causal_inference.py` creado
- ✅ **Configuración**: `config/model_configs/causal_inference_config.yaml` creado
- ✅ **Dependencias**: `requirements.txt` actualizado con `dowhy` y `econml`

#### Próximos Pasos
- [ ] **Instalar dependencias:**
  ```bash
  pip install dowhy econml
  ```

- [ ] **Probar el pipeline:**
  ```bash
  python scripts/run_causal_inference.py
  ```

- [ ] **Añadir datos reales (opcional):**
  - Crear carpeta `data/raw/causal_inference/`
  - Añadir archivos `psm_data.csv` y `did_data.csv` con estructura esperada
  - Ajustar nombres de columnas en `scripts/run_causal_inference.py` según datos reales

- [ ] **Documentar metodología:**
  - Crear `docs/methodology/causal_inference.md`
  - Explicar PSM y DiD
  - Incluir ejemplos de uso

- [ ] **Integrar en Streamlit (opcional):**
  - Crear `app/pages/3_causal_inference.py`
  - Añadir visualizaciones interactivas

### Pilar 4: GenAI & Automatización
- [ ] **RAG (Retrieval Augmented Generation)**
  - [ ] Instalar dependencias: `pip install langchain chromadb`
  - [ ] Implementar `src/models/genai/rag.py`
  - [ ] Crear base de conocimiento de ejemplo (ej: reviews de productos)
  - [ ] Crear script `scripts/run_genai.py`
  - [ ] Añadir configuración en `config/model_configs/genai_config.yaml`
  - [ ] Documentar metodología en `docs/methodology/genai.md`

- [ ] **Generación de Contenido**
  - [ ] Configurar API de OpenAI (requiere API key)
  - [ ] Implementar `src/models/genai/content_generation.py`
  - [ ] Integrar con segmentos RFM del Pilar 1
  - [ ] Crear templates de emails de retención
  - [ ] Añadir manejo de errores y rate limiting

---

## 🔄 Mejoras y Optimizaciones Pendientes

### Testing
- [ ] Crear tests unitarios para módulos del Pilar 1 (Customer Analytics)
- [ ] Crear tests unitarios para módulos del Pilar 2 (Marketing Science)
- [ ] Crear tests de integración para pipelines completos
- [ ] Configurar CI/CD básico (GitHub Actions)

### Documentación
- [ ] Completar documentación de metodología para todos los pilares
- [ ] Crear ejemplos de uso (notebooks o scripts de ejemplo)
- [ ] Añadir diagramas de arquitectura del proyecto
- [ ] Documentar estructura de datos esperada para cada módulo

### Optimizaciones
- [ ] Optimizar cálculos de LTV para datasets grandes
- [ ] Añadir caching para resultados de modelos costosos
- [ ] Implementar paralelización donde sea posible
- [ ] Añadir logging más detallado y métricas de rendimiento

### Integración con Streamlit
- [ ] Integrar Pilar 1 (Customer Analytics) en `app.py`
- [ ] Integrar Pilar 2 (Marketing Science) en `app.py`
- [ ] Crear dashboards interactivos para visualización de resultados
- [ ] Añadir widgets para configuración de modelos desde la UI

---

## 📝 Notas Adicionales

### Dependencias Opcionales
- `lightweight-mmm`: Requiere habilitar rutas largas en Windows (ver arriba)
- `dowhy`: Para inferencia causal (Pilar 3)
- `langchain`, `chromadb`: Para RAG (Pilar 4)
- `openai`: Para generación de contenido (Pilar 4) - requiere API key

### Configuración de Entorno
- Considerar usar un entorno virtual (`venv` o `conda`) para evitar conflictos de dependencias
- Actualizar `requirements.txt` con todas las dependencias nuevas
- Considerar usar `requirements-dev.txt` para dependencias de desarrollo

### Datos
- Verificar que los datasets estén en `data/raw/`
- Añadir scripts de descarga automática de datasets públicos si es posible
- Documentar estructura de datos esperada para cada módulo

---

**Última actualización:** 2025-11-21
**Estado general:** 
- ✅ Pilar 1 (Customer Analytics): Implementado
- ✅ Pilar 2 (Marketing Science): Implementado con PyMC-Marketing (algunas mejoras pendientes)
- ⏳ Pilar 3 (Inferencia Causal): Pendiente
- ⏳ Pilar 4 (GenAI & Automatización): Pendiente


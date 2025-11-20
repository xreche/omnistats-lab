# Tareas Pendientes - OmniStats Lab

## 🔧 Instalación de lightweight-mmm (Marketing Mix Modeling)

### Estado Actual
- ✅ Visual C++ Build Tools instaladas correctamente
- ✅ `matplotlib` compilado exitosamente
- ❌ `lightweight-mmm` no instalado completamente
- ⚠️ TensorFlow falla por rutas de archivos muy largas en Windows

### Problema Identificado
El paquete `tensorflow` (dependencia de `lightweight-mmm`) no se puede instalar debido a que Windows tiene un límite de 260 caracteres para rutas de archivos. Algunos archivos dentro del paquete TensorFlow exceden este límite.

### Solución: Habilitar Rutas Largas en Windows

#### Paso 1: Ejecutar PowerShell como Administrador
1. Presiona `Win + X` y selecciona "Windows PowerShell (Administrador)" o "Terminal (Administrador)"
2. O busca "PowerShell" en el menú de inicio, haz clic derecho y selecciona "Ejecutar como administrador"

#### Paso 2: Habilitar Rutas Largas
Ejecuta el siguiente comando en PowerShell (como Administrador):

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Verificación:**
```powershell
Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled"
```

Debería mostrar `LongPathsEnabled : 1`

#### Paso 3: Reiniciar el Sistema
⚠️ **IMPORTANTE:** Reinicia tu computadora para que los cambios surtan efecto.

#### Paso 4: Instalar lightweight-mmm
Después de reiniciar, abre una nueva terminal y ejecuta:

```bash
pip install lightweight-mmm
```

#### Paso 5: Verificar Instalación
```bash
python -c "from src.models.marketing_science.mmm import LIGHTWEIGHT_MMM_AVAILABLE; print(f'lightweight-mmm disponible: {LIGHTWEIGHT_MMM_AVAILABLE}')"
```

Debería mostrar: `lightweight-mmm disponible: True`

#### Paso 6: Probar el Pipeline Completo
```bash
python scripts/run_marketing_science.py
```

Ahora debería ejecutar el módulo MMM sin problemas.

---

## 📋 Pilares Pendientes de Implementación

### Pilar 3: Inferencia Causal
- [ ] **Propensity Score Matching (PSM)**
  - [ ] Instalar `dowhy`: `pip install dowhy`
  - [ ] Implementar `src/models/causal_inference/psm.py`
  - [ ] Crear script `scripts/run_causal_inference.py`
  - [ ] Añadir configuración en `config/model_configs/causal_inference_config.yaml`
  - [ ] Documentar metodología en `docs/methodology/causal_inference.md`

- [ ] **Difference-in-Differences (DiD)**
  - [ ] Implementar `src/models/causal_inference/did.py`
  - [ ] Integrar en el pipeline de causal inference
  - [ ] Añadir tests y validaciones

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

**Última actualización:** 2025-11-20
**Estado general:** Pilares 1 y 2 implementados. Pilares 3 y 4 pendientes.


# Notas de Instalación - OmniStats Lab

## Dependencias Opcionales

### lightweight-mmm (Marketing Mix Modeling)

**Estado:** Opcional - El código maneja graciosamente su ausencia

**Problema en Windows:**
`lightweight-mmm` requiere `tensorflow`, que puede fallar en instalación debido a **rutas de archivos muy largas** en Windows. Este es un problema conocido de Windows con rutas que exceden 260 caracteres.

**Soluciones:**

**Solución 1: Habilitar rutas largas en Windows (Recomendado)**
1. Abrir PowerShell como Administrador
2. Ejecutar: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`
3. Reiniciar el sistema
4. Luego ejecutar: `pip install lightweight-mmm`

**Solución 2: Usar entorno virtual en ruta corta**
1. Crear un entorno virtual en una ruta corta (ej: `C:\venv\omnistats`)
2. Activar el entorno virtual
3. Instalar: `pip install lightweight-mmm`

**Solución 3: Usar sin lightweight-mmm (Funcional para desarrollo)**
- ✅ El código funciona perfectamente sin esta librería
- ✅ El módulo MMM se saltará automáticamente si no está disponible
- ✅ Todos los demás módulos (Price Elasticity, Attribution) funcionan normalmente
- ✅ El pipeline de Marketing Science se ejecuta correctamente

**Verificación:**
```python
python -c "from src.models.marketing_science.mmm import LIGHTWEIGHT_MMM_AVAILABLE; print(f'lightweight-mmm disponible: {LIGHTWEIGHT_MMM_AVAILABLE}')"
```

**Nota sobre compatibilidad:**
- `lightweight-mmm` requiere `tensorflow>=2.7.2`
- En algunos sistemas Windows, la instalación de TensorFlow puede fallar por rutas largas
- El resto del pipeline (Price Elasticity, Multi-Touch Attribution) **no depende** de `lightweight-mmm` y funciona correctamente

## Dependencias Principales (Ya Instaladas)

✅ **lifetimes** - Para modelos LTV (BG/NBD, Gamma-Gamma)
✅ **statsmodels** - Para econometría (Price Elasticity)
✅ **pandas, numpy, scipy** - Core data science
✅ **scikit-learn** - Machine Learning
✅ **plotly** - Visualización

## Instalación Completa

```bash
# Instalar dependencias principales
pip install -r requirements.txt

# Opcional: lightweight-mmm (requiere Visual C++ Build Tools en Windows)
pip install lightweight-mmm
```


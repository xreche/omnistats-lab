import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, r2_score, 
                             mean_squared_error, mean_absolute_error, silhouette_score)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="OmniStats Lab", layout="wide", page_icon="📊")

# --- FUNCIONES AUXILIARES ---

def generate_continuous_data(dist_type, n, params):
    """Genera datos continuos según distribución."""
    if dist_type == "Normal":
        return np.random.normal(params['loc'], params['scale'], n)
    elif dist_type == "Uniforme":
        return np.random.uniform(params['low'], params['high'], n)
    elif dist_type == "Exponencial":
        return np.random.exponential(params['scale'], n)
    elif dist_type == "Bimodal (No-Normal)":
        # Mezcla de dos normales
        d1 = np.random.normal(params['loc'], params['scale'], int(n/2))
        d2 = np.random.normal(params['loc'] + 4, params['scale'], int(n/2))
        return np.concatenate([d1, d2])
    return np.random.normal(0, 1, n)

def plot_qq(data, title="Q-Q Plot"):
    """Genera un QQ-Plot manual con Plotly."""
    sorted_data = np.sort(data)
    # Cuantiles teóricos (Normal estándar)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='Datos'))
    
    # Línea de ajuste
    slope, intercept, _, _, _ = stats.linregress(theoretical_quantiles, sorted_data)
    line = slope * theoretical_quantiles + intercept
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=line, mode='lines', name='Ajuste Normal', line=dict(color='red', dash='dash')))
    
    fig.update_layout(title=title, xaxis_title="Cuantiles Teóricos", yaxis_title="Cuantiles de la Muestra")
    return fig

def plot_decision_boundary(model, X, y):
    """Visualiza frontera de decisión para modelos 2D."""
    h = .02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Crear figura con go.Figure para poder combinar scatter y contour
    fig = go.Figure()
    
    # Agregar contorno de frontera de decisión primero (va al fondo)
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        showscale=False,
        opacity=0.4,
        colorscale='Viridis',
        contours=dict(showlines=False)
    ))
    
    # Agregar puntos de datos después (quedan visibles encima)
    for class_val in np.unique(y):
        mask = y == class_val
        fig.add_trace(go.Scatter(
            x=X[mask, 0], 
            y=X[mask, 1], 
            mode='markers',
            name=f'Clase {class_val}',
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Frontera de Decisión",
        xaxis_title="X1",
        yaxis_title="X2"
    )
    return fig

# --- SIDEBAR GLOBAL ---
st.sidebar.title("OmniStats Lab 🧪")
seed = st.sidebar.number_input("Semilla (Seed)", value=42, help="Para reproducibilidad")
np.random.seed(seed)

module = st.sidebar.radio("Seleccionar Módulo", 
    ["A. Diagnóstico y Descriptiva", 
     "B. Estadística Inferencial", 
     "C. Aprendizaje Supervisado", 
     "D. Aprendizaje No Supervisado", 
     "E. Deep Learning (MLP)", 
     "F. Reinforcement Learning"])

# --- MÓDULO A: DIAGNÓSTICO ---
if module == "A. Diagnóstico y Descriptiva":
    st.header("Módulo A: Diagnóstico de Datos y Normalidad")
    st.markdown("Antes de inferir, debemos saber si nuestros datos cumplen los supuestos (Normalidad, Homocedasticidad).")
    
    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Tamaño de Muestra (N)", 30, 1000, 200)
        dist_type = st.selectbox("Distribución Generadora", ["Normal", "Uniforme", "Exponencial", "Bimodal (No-Normal)"])
    
    # Parámetros dinámicos
    params = {'loc': 0, 'scale': 1, 'low': -3, 'high': 3}
    if dist_type == "Normal" or dist_type == "Bimodal (No-Normal)":
        params['scale'] = st.sidebar.slider("Desviación Estándar (Sigma)", 0.1, 5.0, 1.0)
    
    data = generate_continuous_data(dist_type, n, params)
    
    # Tests Estadísticos
    shapiro_stat, shapiro_p = stats.shapiro(data)
    
    st.subheader("📊 Resultados del Diagnóstico")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Media", f"{np.mean(data):.2f}")
    metric_col2.metric("Mediana", f"{np.median(data):.2f}")
    metric_col3.metric("Desviación Std", f"{np.std(data):.2f}")
    
    st.divider()
    
    st.write(f"**Test de Shapiro-Wilk (Normalidad):** p-value = `{shapiro_p:.5f}`")
    if shapiro_p > 0.05:
        st.success("✅ No se rechaza la Normalidad (p > 0.05). Sugiere uso de **Pruebas Paramétricas**.")
    else:
        st.error("⚠️ Se rechaza la Normalidad (p < 0.05). Sugiere uso de **Pruebas No Paramétricas**.")
        
    # Visualización
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        fig_hist = px.histogram(data, nbins=30, title="Histograma + KDE", marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)
    with viz_col2:
        fig_qq = plot_qq(data)
        st.plotly_chart(fig_qq, use_container_width=True)

# --- MÓDULO B: INFERENCIAL ---
elif module == "B. Estadística Inferencial":
    st.header("Módulo B: Estadística Inferencial")
    
    tab1, tab2 = st.tabs(["Estimación (Intervalos)", "Contrastes de Hipótesis"])
    
    # --- RAMA 1: ESTIMACIÓN ---
    with tab1:
        st.subheader("Estimación de Parámetros e Intervalos de Confianza")
        estim_type = st.selectbox("¿Qué quieres estimar?", ["Media (Puntual e Intervalo)", "Proporción", "Diferencia de Medias"])
        
        confidence = st.slider("Nivel de Confianza", 0.80, 0.99, 0.95)
        alpha = 1 - confidence
        
        if estim_type == "Media (Puntual e Intervalo)":
            n_est = st.number_input("N", 10, 1000, 50)
            mu_real = st.number_input("Media Real (Simulación)", value=10.0)
            sigma_real = st.number_input("Sigma Real", value=2.0)
            
            data_est = np.random.normal(mu_real, sigma_real, n_est)
            
            # Cálculo IC T-Student
            sem = stats.sem(data_est)
            interval = stats.t.interval(confidence, len(data_est)-1, loc=np.mean(data_est), scale=sem)
            
            st.info(f"**Intervalo de Confianza ({confidence*100}%):** [{interval[0]:.3f}, {interval[1]:.3f}]")
            st.write(f"Media Muestral: {np.mean(data_est):.3f}")
            
            # Gráfico Error Bar
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=["Muestra"], y=[np.mean(data_est)], 
                                     error_y=dict(type='data', array=[interval[1]-np.mean(data_est)], visible=True),
                                     mode='markers', name='Media Estimada'))
            fig.add_hline(y=mu_real, line_dash="dash", annotation_text="Media Poblacional Real")
            st.plotly_chart(fig)
            
        elif estim_type == "Proporción":
            n_prop = st.number_input("Tamaño Muestra", 50, 1000, 100)
            p_real = st.slider("Probabilidad Real de Éxito", 0.0, 1.0, 0.5)
            
            # Generar Bernoulli
            data_bin = np.random.binomial(1, p_real, n_prop)
            p_hat = np.mean(data_bin)
            
            # Intervalo Aproximación Normal
            margin_error = stats.norm.ppf(1 - alpha/2) * np.sqrt((p_hat * (1 - p_hat)) / n_prop)
            ic_low, ic_high = p_hat - margin_error, p_hat + margin_error
            
            st.info(f"**Proporción Estimada:** {p_hat:.3f} | **IC:** [{ic_low:.3f}, {ic_high:.3f}]")
            st.bar_chart(pd.Series(data_bin).value_counts(normalize=True))

        elif estim_type == "Diferencia de Medias":
            col_a, col_b = st.columns(2)
            n1 = col_a.number_input("N Grupo 1", 30, 500, 100)
            n2 = col_b.number_input("N Grupo 2", 30, 500, 100)
            mu1 = col_a.number_input("Media 1", value=10.0)
            mu2 = col_b.number_input("Media 2", value=12.0)
            
            g1 = np.random.normal(mu1, 2, n1)
            g2 = np.random.normal(mu2, 2, n2)
            
            diff_means = np.mean(g1) - np.mean(g2)
            # Pooled SEM (simplificado, asumiendo varianzas iguales para visualización)
            se_diff = np.sqrt(np.var(g1)/n1 + np.var(g2)/n2)
            margin = stats.t.ppf(1 - alpha/2, n1+n2-2) * se_diff
            
            st.write(f"Diferencia de Medias: {diff_means:.3f}")
            st.info(f"IC de la Diferencia: [{diff_means - margin:.3f}, {diff_means + margin:.3f}]")
            if (diff_means - margin) <= 0 <= (diff_means + margin):
                st.warning("El intervalo incluye el 0. No hay diferencia significativa.")
            else:
                st.success("El intervalo NO incluye el 0. Diferencia significativa.")

    # --- RAMA 2: CONTRASTES ---
    with tab2:
        st.subheader("Contrastes de Hipótesis (Deductiva)")
        
        test_family = st.radio("Familia de Pruebas", ["Paramétrica (Medias)", "No Paramétrica (Rangos/Medianas)", "Proporciones (Categóricos)"], horizontal=True)
        
        if test_family == "Paramétrica (Medias)":
            test_type = st.selectbox("Prueba", ["T-Test 1 Muestra", "T-Test 2 Muestras Indep.", "ANOVA (3 Grupos)"])
            
            if test_type == "T-Test 2 Muestras Indep.":
                st.caption("Asume Normalidad y Homocedasticidad.")
                diff_sim = st.slider("Diferencia real entre grupos", 0.0, 5.0, 0.0)
                g1 = np.random.normal(10, 2, 100)
                g2 = np.random.normal(10 + diff_sim, 2, 100)
                
                stat, p = stats.ttest_ind(g1, g2)
                
                df_viz = pd.DataFrame({'Valor': np.concatenate([g1, g2]), 'Grupo': ['A']*100 + ['B']*100})
                st.plotly_chart(px.box(df_viz, x='Grupo', y='Valor', points="all"))
                
                st.metric("P-Value", f"{p:.5f}", delta="Significativo" if p < 0.05 else "No Signif.", delta_color="inverse")
            
            elif test_type == "ANOVA (3 Grupos)":
                means = st.multiselect("Medias de los 3 grupos", [10, 12, 15, 20], default=[10, 10, 12])
                if len(means) < 3: st.warning("Selecciona 3 medias para el ejemplo")
                else:
                    grps = [np.random.normal(m, 2, 50) for m in means[:3]]
                    stat, p = stats.f_oneway(*grps)
                    st.metric("ANOVA P-Value", f"{p:.5f}")
                    
                    data_anova = []
                    for i, g in enumerate(grps):
                        for val in g: data_anova.append({'Grupo': f'G{i+1}', 'Valor': val})
                    st.plotly_chart(px.box(pd.DataFrame(data_anova), x='Grupo', y='Valor'))

        elif test_family == "No Paramétrica (Rangos/Medianas)":
            st.markdown("**Uso:** Cuando los datos NO son normales o tienen muchos outliers.")
            test_type = st.selectbox("Prueba", ["Mann-Whitney U (2 Muestras)", "Kruskal-Wallis (>2 Muestras)"])
            
            if test_type == "Mann-Whitney U (2 Muestras)":
                shift = st.slider("Desplazamiento Mediana G2", 0.0, 3.0, 0.5)
                # Generamos datos exponenciales (no normales)
                g1 = np.random.exponential(1, 100)
                g2 = np.random.exponential(1, 100) + shift
                
                stat, p = stats.mannwhitneyu(g1, g2)
                st.metric("Mann-Whitney P-Value", f"{p:.5f}")
                
                df_viz = pd.DataFrame({'Valor': np.concatenate([g1, g2]), 'Grupo': ['A']*100 + ['B']*100})
                st.plotly_chart(px.box(df_viz, x='Grupo', y='Valor', title="Datos No Normales (Exponenciales)"))

        elif test_family == "Proporciones (Categóricos)":
            st.markdown("**Prueba de Independencia (Chi-Cuadrado)**")
            st.write("Simulando tabla de contingencia 2x2 (Ej: Fuma vs. Cáncer)")
            
            p_g1 = st.slider("Prob. Evento Grupo A", 0.1, 0.9, 0.3)
            p_g2 = st.slider("Prob. Evento Grupo B", 0.1, 0.9, 0.3)
            
            n = 200
            g1 = np.random.choice(['Si', 'No'], n, p=[p_g1, 1-p_g1])
            g2 = np.random.choice(['Si', 'No'], n, p=[p_g2, 1-p_g2]) # Simulación simplificada para inputs
            
            # Creamos datos más estructurados para controlar la relación
            # Método directo: Crear la tabla
            obs = np.array([[int(n*p_g1), int(n*(1-p_g1))], [int(n*p_g2), int(n*(1-p_g2))]])
            
            chi2, p, dof, ex = stats.chi2_contingency(obs)
            
            st.write("Tabla de Contingencia Generada:")
            st.dataframe(pd.DataFrame(obs, columns=['Si', 'No'], index=['Grupo A', 'Grupo B']))
            
            st.metric("Chi2 P-Value", f"{p:.5f}")
            if p < 0.05: st.success("Existe dependencia significativa entre las variables.")
            else: st.warning("Variables independientes (No hay relación significativa).")

# --- MÓDULO C: SUPERVISADO ---
elif module == "C. Aprendizaje Supervisado":
    st.header("Aprendizaje Supervisado: Regresión y Clasificación")
    type_ml = st.radio("Tipo de Problema", ["Clasificación", "Regresión"])
    
    noise = st.slider("Nivel de Ruido", 0.0, 1.0, 0.1)
    
    if type_ml == "Clasificación":
        st.markdown("""
        ### 📝 Configuración del Problema de Clasificación
        
        En este módulo, vamos a predecir la **clase (categoría)** de cada punto usando sus **características (X1, X2)**.
        - **Variables Predictoras (X1, X2)**: Características que usamos para hacer predicciones
        - **Variable Objetivo (Y)**: Clase a la que pertenece cada punto (0 o 1)
        
        El modelo aprenderá a separar las clases basándose en las características.
        """)
        
        st.divider()
        
        # Configuración de datos
        st.subheader("⚙️ Configuración de Datos")
        
        col_data1, col_data2 = st.columns(2)
        
        with col_data1:
            dataset_name = st.selectbox(
                "Tipo de Dataset", 
                ["Moons (Lunas)", "Circles (Círculos)"],
                help="Moons: Datos con forma de dos lunas entrelazadas. Circles: Datos con forma de círculos concéntricos."
            )
            n_samples = st.number_input("Número de Muestras", min_value=100, max_value=1000, value=300, step=50,
                                       help="Cantidad de puntos de datos a generar")
        
        with col_data2:
            noise_level = st.slider("Nivel de Ruido", 0.0, 1.0, noise, step=0.05,
                                   help="Cantidad de ruido/solapamiento entre las clases. Mayor ruido = más difícil de separar")
            if dataset_name == "Circles (Círculos)":
                factor = st.slider("Factor de Separación", 0.1, 0.9, 0.5, step=0.1,
                                  help="Distancia entre los círculos. Menor = más difícil de separar")
        
        # Generar datos
        if dataset_name == "Moons (Lunas)":
            X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=seed)
            dataset_desc = "Dos lunas entrelazadas"
        else:
            X, y = make_circles(n_samples=n_samples, noise=noise_level, factor=factor, random_state=seed)
            dataset_desc = "Círculos concéntricos"
        
        # Mostrar información sobre los datos
        st.info(f"""
        **📊 Información del Dataset:**
        - **Tipo**: {dataset_desc}
        - **Variables Predictoras**: X1 (primera coordenada), X2 (segunda coordenada)
        - **Variable Objetivo**: Clase (0 o 1)
        - **Número de muestras**: {n_samples} ({np.sum(y==0)} de clase 0, {np.sum(y==1)} de clase 1)
        - **Nivel de ruido**: {noise_level:.2f}
        """)
        
        st.divider()
        
        # Configuración del modelo
        st.subheader("🤖 Configuración del Modelo")
        
        model_name = st.selectbox(
            "Algoritmo de Clasificación", 
            ["Regresión Logística", "Árbol de Decisión", "Red Neuronal Simple"],
            help="Regresión Logística: Modelo lineal. Árbol de Decisión: Modelo no lineal basado en reglas. Red Neuronal: Modelo no lineal con capas ocultas."
        )
        
        # Normalizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Configurar modelo según selección
        if model_name == "Regresión Logística":
            model = LogisticRegression(random_state=seed, max_iter=1000)
            model_desc = "Modelo lineal que separa las clases con una línea/plano"
        elif model_name == "Árbol de Decisión":
            max_depth = st.slider("Profundidad Máxima del Árbol", 1, 10, 5,
                                 help="Profundidad máxima del árbol. Mayor profundidad = modelo más complejo")
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
            model_desc = f"Modelo no lineal basado en reglas de decisión (profundidad: {max_depth})"
        else:
            hidden_units = st.slider("Neuronas en Capa Oculta", 5, 50, 20, step=5,
                                    help="Número de neuronas en la capa oculta")
            model = MLPClassifier(hidden_layer_sizes=(hidden_units,), max_iter=500, alpha=0.01, random_state=seed)
            model_desc = f"Red neuronal con {hidden_units} neuronas en la capa oculta"
        
        st.caption(f"**Modelo seleccionado:** {model_desc}")
        
        # Entrenar modelo
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        
        # Métricas de Clasificación
        st.subheader("📊 Métricas de Rendimiento")
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Mostrar métricas en columnas
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Accuracy", f"{accuracy:.4f}")
        metric_col2.metric("Precision", f"{precision:.4f}")
        metric_col3.metric("Recall", f"{recall:.4f}")
        metric_col4.metric("F1-Score", f"{f1:.4f}")
        
        # Matriz de Confusión
        st.subheader("📈 Matriz de Confusión")
        cm = confusion_matrix(y, y_pred)
        
        # Visualizar matriz de confusión con Plotly
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Pred. {i}' for i in range(len(np.unique(y)))],
            y=[f'Real {i}' for i in range(len(np.unique(y)))],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            hoverongaps=False
        ))
        fig_cm.update_layout(title="Matriz de Confusión", width=500, height=500)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Reporte de clasificación
        with st.expander("Ver Reporte Detallado de Clasificación"):
            report = classification_report(y, y_pred, output_dict=True)
            st.json(report)
        
        st.divider()
        st.subheader("🎯 Visualización de la Frontera de Decisión")
        st.caption("La frontera muestra cómo el modelo separa las dos clases basándose en las características X1 y X2")
        st.plotly_chart(plot_decision_boundary(model, X_scaled, y))
        
    else: # Regresión
        st.markdown("""
        ### 📝 Configuración del Problema de Regresión
        
        En este módulo, vamos a predecir una **variable dependiente (Y)** usando una **variable independiente (X)**.
        - **Variable Predictora (X)**: Variable que usamos para hacer predicciones
        - **Variable Objetivo (Y)**: Variable que queremos predecir
        
        Puedes configurar la relación real entre X e Y, y luego ver cómo el modelo intenta aprenderla.
        """)
        
        st.divider()
        
        # Configuración de datos
        st.subheader("⚙️ Configuración de Datos")
        
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            n_samples = st.number_input("Número de Muestras", min_value=50, max_value=1000, value=100, step=10,
                                       help="Cantidad de puntos de datos a generar")
            x_min = st.number_input("Valor Mínimo de X", value=-3.0, step=0.5,
                                   help="Valor mínimo de la variable predictora")
        
        with col_config2:
            x_max = st.number_input("Valor Máximo de X", value=3.0, step=0.5,
                                   help="Valor máximo de la variable predictora")
            noise_level = st.slider("Nivel de Ruido", 0.0, 10.0, noise*5, step=0.5,
                                    help="Cantidad de ruido aleatorio en los datos")
        
        with col_config3:
            st.markdown("**Función Real (Relación X → Y)**")
            st.caption("Configura los coeficientes de la función polinomial que genera Y")
            
            coef_cubic = st.number_input("Coeficiente X³", value=0.5, step=0.1, format="%.2f")
            coef_quad = st.number_input("Coeficiente X²", value=-2.0, step=0.1, format="%.2f")
            coef_linear = st.number_input("Coeficiente X¹", value=0.0, step=0.1, format="%.2f")
            coef_const = st.number_input("Término Constante", value=0.0, step=0.1, format="%.2f")
        
        # Generar datos
        X = np.linspace(x_min, x_max, n_samples).reshape(-1, 1)
        
        # Función real: y = coef_cubic * X³ + coef_quad * X² + coef_linear * X + coef_const + ruido
        y_true = coef_cubic * X**3 + coef_quad * X**2 + coef_linear * X + coef_const
        y = y_true + np.random.normal(0, noise_level, (n_samples, 1))
        y = y.ravel()
        
        # Mostrar información sobre los datos
        st.info(f"""
        **📊 Información del Dataset:**
        - **Variable Predictora (X)**: Rango de {x_min} a {x_max} con {n_samples} muestras
        - **Variable Objetivo (Y)**: Generada por la función Y = {coef_cubic:.2f}X³ + {coef_quad:.2f}X² + {coef_linear:.2f}X + {coef_const:.2f} + ruido
        - **Ruido**: Desviación estándar = {noise_level:.2f}
        """)
        
        st.divider()
        
        # Configuración del modelo
        st.subheader("🤖 Configuración del Modelo")
        degree = st.slider("Grado del Polinomio del Modelo", 1, 15, 1,
                          help="Complejidad del modelo. Grado 1 = línea recta, grado 2 = parábola, etc.")
        
        st.caption(f"El modelo intentará ajustar un polinomio de grado {degree} a los datos.")
        
        # Pipeline manual polinomial
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        # Mostrar información del modelo
        with st.expander("📋 Ver Coeficientes del Modelo Ajustado"):
            coefs = model.coef_
            intercept = model.intercept_
            st.write("**Ecuación del Modelo Ajustado:**")
            eq_parts = []
            if abs(intercept) > 1e-6:
                eq_parts.append(f"{intercept:.4f}")
            for i, coef in enumerate(coefs[1:], 1):  # Saltar el término constante (ya está en intercept)
                if abs(coef) > 1e-6:
                    if i == 1:
                        eq_parts.append(f"{coef:.4f}X")
                    elif i == 2:
                        eq_parts.append(f"{coef:.4f}X²")
                    elif i == 3:
                        eq_parts.append(f"{coef:.4f}X³")
                    else:
                        eq_parts.append(f"{coef:.4f}X^{i}")
            equation = "Y = " + " + ".join(eq_parts) if eq_parts else "Y = 0"
            st.code(equation)
            st.write(f"**Número de parámetros:** {len(coefs) + 1}")
        
        # Métricas de Regresión
        st.subheader("📊 Métricas de Rendimiento")
        
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Mostrar métricas en columnas
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("R² (Coef. Determinación)", f"{r2:.4f}")
        metric_col2.metric("MSE", f"{mse:.4f}")
        metric_col3.metric("MAE", f"{mae:.4f}")
        metric_col4.metric("RMSE", f"{rmse:.4f}")
        
        # Interpretación de R²
        if r2 > 0.9:
            st.success(f"✅ Excelente ajuste (R² = {r2:.4f}). El modelo explica más del 90% de la varianza.")
        elif r2 > 0.7:
            st.info(f"ℹ️ Buen ajuste (R² = {r2:.4f}). El modelo explica más del 70% de la varianza.")
        elif r2 > 0.5:
            st.warning(f"⚠️ Ajuste moderado (R² = {r2:.4f}). El modelo explica más del 50% de la varianza.")
        else:
            st.error(f"❌ Ajuste pobre (R² = {r2:.4f}). El modelo explica menos del 50% de la varianza.")
        
        st.divider()
        st.subheader("📈 Visualización del Modelo")
        
        # Crear figura con múltiples trazas
        fig = go.Figure()
        
        # Datos observados (con ruido)
        fig.add_trace(go.Scatter(
            x=X.ravel(), 
            y=y, 
            mode='markers', 
            name='Datos Observados (Y con ruido)', 
            marker=dict(color='blue', size=6, opacity=0.6),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        # Función real (sin ruido) - si el usuario quiere verla
        if st.checkbox("Mostrar función real (sin ruido)", value=True):
            fig.add_trace(go.Scatter(
                x=X.ravel(), 
                y=y_true.ravel(), 
                mode='lines', 
                name='Función Real (Y verdadera)', 
                line=dict(color='green', width=2, dash='dash'),
                hovertemplate='X: %{x:.2f}<br>Y real: %{y:.2f}<extra></extra>'
            ))
        
        # Predicción del modelo
        fig.add_trace(go.Scatter(
            x=X.ravel(), 
            y=y_pred, 
            mode='lines', 
            name=f'Predicción del Modelo (Grado {degree})', 
            line=dict(color='red', width=3),
            hovertemplate='X: %{x:.2f}<br>Y predicha: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Regresión Polinomial: Predicción de Y usando X (Modelo Grado {degree})",
            xaxis_title="Variable Predictora (X)",
            yaxis_title="Variable Objetivo (Y)",
            hovermode='closest',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de datos (opcional)
        with st.expander("📊 Ver Tabla de Datos"):
            df_display = pd.DataFrame({
                'X (Variable Predictora)': X.ravel(),
                'Y Observado (con ruido)': y,
                'Y Real (sin ruido)': y_true.ravel(),
                'Y Predicho (Modelo)': y_pred,
                'Error (Y Observado - Y Predicho)': y - y_pred
            })
            st.dataframe(df_display.round(3), use_container_width=True)

# --- MÓDULO D: NO SUPERVISADO ---
elif module == "D. Aprendizaje No Supervisado":
    st.header("Clustering: K-Means vs DBSCAN")
    
    n_centers = st.slider("Número Real de Clusters (Generación)", 2, 6, 3)
    cluster_std = st.slider("Dispersión de Clusters", 0.1, 2.0, 0.5)
    X, _ = make_blobs(n_samples=300, centers=n_centers, cluster_std=cluster_std, random_state=seed)
    
    algo = st.selectbox("Algoritmo", ["K-Means", "DBSCAN"])
    
    if algo == "K-Means":
        k = st.slider("K (Clusters a buscar)", 1, 8, 3)
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        y_pred = model.fit_predict(X)
        
        # Métricas para K-Means
        st.subheader("📊 Métricas de Rendimiento")
        
        inertia = model.inertia_
        silhouette = silhouette_score(X, y_pred)
        n_clusters_found = len(np.unique(y_pred))
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Inertia (WCSS)", f"{inertia:.2f}", 
                           help="Suma de distancias al cuadrado de las muestras a su centroide más cercano. Menor es mejor.")
        metric_col2.metric("Silhouette Score", f"{silhouette:.4f}",
                          help="Mide qué tan bien separados están los clusters. Rango: -1 a 1. Mayor es mejor.")
        metric_col3.metric("Clusters Encontrados", f"{n_clusters_found}", 
                          help="Número de clusters identificados por el algoritmo.")
        
        # Interpretación de Silhouette
        if silhouette > 0.5:
            st.success(f"✅ Excelente separación de clusters (Silhouette = {silhouette:.4f})")
        elif silhouette > 0.3:
            st.info(f"ℹ️ Buena separación de clusters (Silhouette = {silhouette:.4f})")
        elif silhouette > 0:
            st.warning(f"⚠️ Separación débil (Silhouette = {silhouette:.4f})")
        else:
            st.error(f"❌ Clusters mal definidos (Silhouette = {silhouette:.4f})")
        
    else:
        eps = st.slider("Epsilon (Radio vecindad)", 0.1, 2.0, 0.5)
        min_samples = st.slider("Min Samples", 2, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = model.fit_predict(X)
        
        # Métricas para DBSCAN
        st.subheader("📊 Métricas de Rendimiento")
        
        n_clusters_found = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        n_noise = list(y_pred).count(-1)
        silhouette = silhouette_score(X, y_pred) if n_clusters_found > 1 else -1
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Clusters Encontrados", f"{n_clusters_found}",
                          help="Número de clusters densos identificados (excluyendo ruido)")
        metric_col2.metric("Puntos de Ruido", f"{n_noise}",
                         help="Puntos clasificados como outliers/noise (-1)")
        metric_col3.metric("Silhouette Score", f"{silhouette:.4f}" if silhouette >= 0 else "N/A",
                          help="Mide qué tan bien separados están los clusters")
        
        if n_clusters_found == 0:
            st.error("❌ No se encontraron clusters. Ajusta los parámetros (eps, min_samples).")
        elif n_noise > len(y_pred) * 0.5:
            st.warning(f"⚠️ Muchos puntos clasificados como ruido ({n_noise}/{len(y_pred)}). Considera aumentar eps o reducir min_samples.")
        else:
            st.success(f"✅ Se encontraron {n_clusters_found} clusters con {n_noise} puntos de ruido.")
    
    st.divider()
    st.subheader("📈 Visualización de Clusters")
    fig = px.scatter(x=X[:,0], y=X[:,1], color=y_pred.astype(str), 
                     title=f"Resultado {algo}",
                     labels={'x': 'X1', 'y': 'X2'},
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

# --- MÓDULO E: DEEP LEARNING ---
elif module == "E. Deep Learning (MLP)":
    st.header("Deep Learning: Entrenamiento en Vivo")
    st.markdown("Entrenando un Perceptrón Multicapa (MLP) en el dataset 'Moons'.")
    
    col1, col2 = st.columns(2)
    lr = col1.selectbox("Learning Rate", [0.001, 0.01, 0.1])
    hidden_units = col2.slider("Neuronas en capa oculta", 5, 100, 20)
    
    if st.button("Entrenar Red"):
        X, y = make_moons(n_samples=500, noise=0.2, random_state=seed)
        X = StandardScaler().fit_transform(X)
        
        # Usamos partial_fit para simular épocas? No, sklearn MLP guarda loss_curve_
        mlp = MLPClassifier(hidden_layer_sizes=(hidden_units,), learning_rate_init=lr, max_iter=500, random_state=seed)
        mlp.fit(X, y)
        
        y_pred = mlp.predict(X)
        
        st.success(f"✅ Entrenamiento finalizado. Iteraciones: {mlp.n_iter_}")
        
        # Métricas de Clasificación
        st.subheader("📊 Métricas de Rendimiento")
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Mostrar métricas en columnas
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Accuracy", f"{accuracy:.4f}")
        metric_col2.metric("Precision", f"{precision:.4f}")
        metric_col3.metric("Recall", f"{recall:.4f}")
        metric_col4.metric("F1-Score", f"{f1:.4f}")
        
        # Matriz de Confusión
        st.subheader("📈 Matriz de Confusión")
        cm = confusion_matrix(y, y_pred)
        
        # Visualizar matriz de confusión con Plotly
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Pred. {i}' for i in range(len(np.unique(y)))],
            y=[f'Real {i}' for i in range(len(np.unique(y)))],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            hoverongaps=False
        ))
        fig_cm.update_layout(title="Matriz de Confusión", width=500, height=500)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Reporte de clasificación
        with st.expander("Ver Reporte Detallado de Clasificación"):
            report = classification_report(y, y_pred, output_dict=True)
            st.json(report)
        
        st.divider()
        
        # Curva de Loss
        st.subheader("📉 Curva de Aprendizaje")
        fig_loss = px.line(y=mlp.loss_curve_, 
                          labels={'x': 'Épocas', 'y': 'Loss'}, 
                          title="Curva de Aprendizaje (Loss)",
                          markers=True)
        fig_loss.update_layout(showlegend=False)
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Frontera
        st.subheader("🎯 Visualización de la Frontera de Decisión")
        st.plotly_chart(plot_decision_boundary(mlp, X, y))

# --- MÓDULO F: REINFORCEMENT LEARNING ---
elif module == "F. Reinforcement Learning":
    st.header("Reinforcement Learning: Multi-Armed Bandit")
    st.markdown("Problema: Tenemos N tragaperras, cada una con una probabilidad de premio desconocida. El agente debe aprender cuál es la mejor.")
    
    n_arms = st.slider("Número de Brazos (Slots)", 2, 10, 5)
    epsilon = st.slider("Epsilon (Exploración)", 0.0, 1.0, 0.1)
    steps = st.slider("Pasos de Simulación", 100, 2000, 1000)
    
    if st.button("Simular Agente"):
        # Probabilidades reales ocultas
        true_probs = np.random.rand(n_arms)
        
        # Inicialización
        q_values = np.zeros(n_arms)
        arm_counts = np.zeros(n_arms)
        rewards = []
        avg_rewards = []
        actions_taken = []  # Rastrear acciones tomadas
        
        cumulative_reward = 0
        optimal_arm = np.argmax(true_probs)
        
        for i in range(steps):
            # Epsilon-Greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(n_arms) # Explorar
            else:
                action = np.argmax(q_values) # Explotar
            
            actions_taken.append(action)
            
            # Simular recompensa (Bernoulli)
            reward = 1 if np.random.rand() < true_probs[action] else 0
            
            # Actualizar Q-Value
            arm_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / arm_counts[action]
            
            cumulative_reward += reward
            avg_rewards.append(cumulative_reward / (i + 1))
        
        # Calcular métricas adicionales
        optimal_reward_rate = true_probs[optimal_arm]
        final_avg_reward = avg_rewards[-1]
        optimal_action_count = sum(1 for a in actions_taken if a == optimal_arm)
        optimal_action_rate = optimal_action_count / steps
        
        # Calcular regret (diferencia entre recompensa óptima y obtenida)
        max_possible_reward = optimal_reward_rate * steps
        actual_reward = cumulative_reward
        regret = max_possible_reward - actual_reward
        regret_rate = regret / steps
        
        st.subheader("📊 Métricas de Rendimiento")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Recompensa Total", f"{cumulative_reward}/{steps}",
                          help="Recompensas obtenidas vs. pasos totales")
        metric_col2.metric("Recompensa Promedio Final", f"{final_avg_reward:.4f}",
                         help="Recompensa promedio al final del entrenamiento")
        metric_col3.metric("Tasa Óptima Real", f"{optimal_reward_rate:.4f}",
                          help="Mejor tasa de recompensa posible (brazo óptimo)")
        metric_col4.metric("Regret Total", f"{regret:.2f}",
                          help="Diferencia entre recompensa óptima y obtenida")
        
        metric_col5, metric_col6, metric_col7 = st.columns(3)
        metric_col5.metric("Tasa de Acción Óptima", f"{optimal_action_rate:.2%}",
                          help="Porcentaje de veces que se eligió el brazo óptimo")
        metric_col6.metric("Regret Rate", f"{regret_rate:.4f}",
                          help="Regret promedio por paso")
        metric_col7.metric("Eficiencia", f"{(1 - regret_rate/optimal_reward_rate):.2%}",
                          help="Eficiencia relativa al óptimo")
        
        # Interpretación
        if final_avg_reward >= optimal_reward_rate * 0.9:
            st.success(f"✅ Excelente rendimiento. El agente aprendió casi tan bien como el óptimo ({final_avg_reward:.4f} vs {optimal_reward_rate:.4f})")
        elif final_avg_reward >= optimal_reward_rate * 0.7:
            st.info(f"ℹ️ Buen rendimiento. El agente aprendió razonablemente bien ({final_avg_reward:.4f} vs {optimal_reward_rate:.4f})")
        else:
            st.warning(f"⚠️ Rendimiento subóptimo. El agente necesita más exploración o pasos ({final_avg_reward:.4f} vs {optimal_reward_rate:.4f})")
        
        st.divider()
        
        # Visualizaciones
        st.subheader("📈 Visualizaciones")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Probabilidades Reales (Ocultas):**")
            fig_probs = go.Figure(data=[go.Bar(
                x=[f'Brazo {i+1}' for i in range(n_arms)],
                y=true_probs,
                marker_color='lightblue',
                text=[f'{p:.3f}' for p in true_probs],
                textposition='auto'
            )])
            fig_probs.update_layout(title="Probabilidades Reales de Recompensa", 
                                   yaxis_title="Probabilidad", 
                                   yaxis_range=[0, 1])
            st.plotly_chart(fig_probs, use_container_width=True)
        
        with col2:
            st.write("**Valores Estimados por el Agente (Q-Values):**")
            fig_q = go.Figure(data=[go.Bar(
                x=[f'Brazo {i+1}' for i in range(n_arms)],
                y=q_values,
                marker_color='lightcoral',
                text=[f'{q:.3f}' for q in q_values],
                textposition='auto'
            )])
            fig_q.update_layout(title="Q-Values Estimados", 
                               yaxis_title="Q-Value", 
                               yaxis_range=[0, 1])
            st.plotly_chart(fig_q, use_container_width=True)
        
        st.subheader("📉 Convergencia del Aprendizaje")
        fig_conv = px.line(y=avg_rewards, 
                          labels={'x': 'Pasos', 'y': 'Recompensa Promedio'}, 
                          title="Convergencia del Aprendizaje",
                          markers=True)
        fig_conv.add_hline(y=optimal_reward_rate, line_dash="dash", 
                          annotation_text=f"Óptimo ({optimal_reward_rate:.4f})",
                          line_color="red")
        fig_conv.update_layout(showlegend=False)
        st.plotly_chart(fig_conv, use_container_width=True)
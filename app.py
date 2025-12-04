import streamlit as st
from bigml.ensemble import Ensemble
from bigml.api import BigML

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="IA Diagnóstico TM", page_icon="🩸", layout="centered")

# --- 1. CONEXIÓN AL CEREBRO REAL ---
@st.cache_resource
def cargar_modelo():
    # Tus credenciales
    api = BigML("mahumada1210", 
                "71c52a8bf7eea6966465f8fced5a0809a4e946d3", 
                domain="bigml.io")
    # Tu Ensamble de Diabetes
    ensemble = Ensemble('ensemble/692c79358c54fd5f980b9091', api=api)
    return ensemble

# Estado de conexión en barra lateral
try:
    ensemble = cargar_modelo()
    st.sidebar.success("✅ Conectado a BigML")
except Exception as e:
    st.error(f"❌ Error de conexión: {e}")

# --- 2. INTERFAZ VISUAL ---
st.title("🩸 Sistema de Soporte Clínico")
st.markdown("### 🤖 Diagnóstico Asistido por IA")
st.info("Ingrese los parámetros. Si no tiene un dato opcional, déjelo desactivado.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Datos Críticos")
    # Valores por defecto para prueba rápida
    glucose = st.number_input("Glucosa (mg/dL)", min_value=0, max_value=600, value=150)
    bmi = st.number_input("BMI (Índice de Masa)", min_value=10.0, max_value=60.0, value=19.0, step=0.1)
    age = st.number_input("Edad (Años)", min_value=1, max_value=120, value=55)

with col2:
    st.subheader("Datos Secundarios")
    
    # Checkbox: Si no se marca, el dato NO existe para el modelo (Evita ruido)
    usar_insulina = st.checkbox("Tengo dato de Insulina", value=False)
    if usar_insulina:
        insulin = st.number_input("Insulina (mu U/ml)", value=100)
    else:
        insulin = None

    usar_presion = st.checkbox("Tengo dato de Presión", value=False)
    if usar_presion:
        blood_pressure = st.number_input("Presión Arterial", value=75)
    else:
        blood_pressure = None

# Empaquetamos SOLO lo que existe
input_data = {
    "Glucose": glucose,
    "BMI": bmi,
    "Age": age
}

if insulin is not None:
    input_data["Insulin"] = insulin
if blood_pressure is not None:
    input_data["BloodPressure"] = blood_pressure

# --- 3. PREDICCIÓN ---
if st.button("CALCULAR RIESGO AHORA 🚀", type="primary"):
    
    with st.spinner('Consultando a BigML...'):
        # Predicción
        resultado = ensemble.predict(input_data)
        etiqueta = resultado['prediction']
        confianza = resultado['probability']
        
        # Cálculo de riesgo
        if etiqueta == "TRUE":
            prob_enfermedad = confianza
        else:
            prob_enfermedad = 1.0 - confianza
            
        porcentaje = prob_enfermedad * 100
        
        # --- RESULTADOS ---
        st.divider()
        
        # UMBRAL DEL 8% (El que nos funcionó bien)
        UMBRAL = 0.08
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label="Riesgo Calculado", value=f"{porcentaje:.2f}%")
        
        with col_res2:
            if prob_enfermedad > UMBRAL:
                st.error("🚨 ESTADO: ALERTA DE RIESGO")
                st.markdown(f"**Supera el umbral de seguridad del {UMBRAL*100}%.**")
                st.markdown("👉 **ACCIÓN:** Repetir examen / Confirmar con clínica.")
            else:
                st.success("🟢 ESTADO: NEGATIVO / BAJO RIESGO")
                st.markdown("**Paciente dentro de rangos seguros.**")
                st.markdown("👉 **ACCIÓN:** Control de rutina.")

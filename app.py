import streamlit as st
from bigml.ensemble import Ensemble
from bigml.api import BigML

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="IA Diabetes TM", page_icon="🩸", layout="centered")

# --- 1. CONEXIÓN AL CEREBRO REAL ---
# Usamos @st.cache para que no cargue el modelo cada vez que haces clic, sino que lo guarde en memoria.
@st.cache_resource
def cargar_modelo():
    # Tus credenciales reales
    api = BigML("mahumada1210", 
                "71c52a8bf7eea6966465f8fced5a0809a4e946d3", 
                domain="bigml.io")
    # Tu Ensamble de Diabetes
    ensemble = Ensemble('ensemble/692c79358c54fd5f980b9091', api=api)
    return ensemble

# Carga inicial con mensaje de estado
try:
    ensemble = cargar_modelo()
    st.success("✅ Sistema Conectado al Analizador BigML (Nube)")
except Exception as e:
    st.error(f"❌ Error de conexión: {e}")

# --- 2. INTERFAZ VISUAL (Entrada de Datos) ---
st.title("🩸 Diagnóstico IA: Diabetes")
st.markdown("Herramienta de soporte clínico basada en Ensamble de Árboles de Decisión.")

st.markdown("### Ingreso de Datos del Paciente")

# Creamos las casillas para meter los números
col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input("Glucosa (mg/dL)", min_value=0, max_value=500, value=105)
    bmi = st.number_input("BMI (Índice Masa)", min_value=10.0, max_value=60.0, value=26.0)
    age = st.number_input("Edad (Años)", min_value=1, max_value=100, value=45)

with col2:
    # Variables opcionales pero útiles
    insulin = st.number_input("Insulina (mu U/ml)", min_value=0, value=100)
    blood_pressure = st.number_input("Presión Arterial", min_value=0, value=75)
    # Aquí podrías agregar más si quieres

# Empaquetamos los datos en un diccionario, igual que en Colab
input_data = {
    "Glucose": glucose,
    "BMI": bmi,
    "Age": age,
    "Insulin": insulin,
    "BloodPressure": blood_pressure
}

# --- 3. BOTÓN Y LÓGICA DE PREDICCIÓN ---
if st.button("PROCESAR MUESTRA 🧬", type="primary"):
    
    with st.spinner('Analizando patrones...'):
        # Consulta real a BigML
        prediccion = ensemble.predict(input_data)
        
        # Extracción de datos crudos
        etiqueta = prediccion['prediction']
        confianza = prediccion['probability']
        
        # Cálculo del riesgo real (Lógica validada)
        if etiqueta == "TRUE":
            prob_enfermedad = confianza
        else:
            prob_enfermedad = 1.0 - confianza
            
        porcentaje = prob_enfermedad * 100
        
        # --- 4. RESULTADO VISUAL ---
        st.divider()
        st.metric(label="Probabilidad de Patología", value=f"{porcentaje:.2f}%")
        
        # TU REGLA DEL 8%
        UMBRAL = 0.08
        
        if prob_enfermedad > UMBRAL:
            st.error(f"🔴 ALERTA: RIESGO DETECTADO")
            st.write(f"El riesgo supera el umbral de seguridad del {UMBRAL*100}%.")
            st.warning("👉 Acción: Repetir examen / Confirmación clínica.")
        else:
            st.success(f"🟢 NEGATIVO / BAJO RIESGO")
            st.write("Paciente dentro de rangos seguros según el modelo.")
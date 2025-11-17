import math

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# ---------------------------------------------------------
# CONFIGURACIÓN BÁSICA DE LA APP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Mapa 3D de peatones",
    layout="wide",
)

st.title("Mapa 3D de peatones por zona")
st.markdown(
    """
    Visualización 3D de la intensidad de peatones por zona (datos sintéticos).

    Usa el *slider* para elegir la **franja horaria** del día.
    Los colores siguen este gradiente aproximado:

    - Azul oscuro → zonas de menor tráfico  
    - Turquesa / verde → tráfico medio  
    - Amarillo → tráfico alto  
    - Naranja-rojo → picos máximos de tráfico
    """
)

# ---------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # CSV sintético
    df = pd.read_csv("df_synthetic_pedestrians.csv")

    # "hora" (ej. "10:00") -> entero 0–23
    def parse_hour(s):
        try:
            return int(str(s).split(":")[0])
        except Exception:
            return None

    df["hour"] = df["hora"].apply(parse_hour)

    # Renombramos para trabajar más cómodo
    df.rename(
        columns={
            "latitude": "lat",
            "longitude": "lon",
            "noisy_peatones_correlated": "peatones",
        },
        inplace=True,
    )

    # Limpiamos
    df = df.dropna(subset=["lat", "lon", "hour", "peatones"])

    return df


df = load_data()

# ---------------------------------------------------------
# SLIDER DE FRANJA HORARIA
# ---------------------------------------------------------
st.subheader("Filtro temporal")

min_hour = int(df["hour"].min())
max_hour = int(df["hour"].max())

# Valores por defecto razonables dentro del rango
default_start = min(max(min_hour, 8), max_hour)
default_end = max(min(max_hour, 20), min_hour)

start_hour, end_hour = st.slider(
    "Selecciona la franja horaria (horas del día)",
    min_value=min_hour,
    max_value=max_hour,
    value=(default_start, default_end),
    step=1,
)

st.markdown(
    f"Mostrando la **intensidad media de peatones** entre "
    f"**{start_hour}:00** y **{end_hour}:00**."
)

mask = (df["hour"] >= start_hour) & (df["hour"] <= end_hour)
df_filtered = df[mask].copy()

if df_filtered.empty:
    st.warning("No hay datos para esa franja horaria.")
    st.stop()

# ---------------------------------------------------------
# AGREGACIÓN POR ZONA (DISTRITO + LAT/LON)
# ---------------------------------------------------------
agg = (
    df_filtered.groupby(
        ["distrito", "lat", "lon"], as_index=False
    )["peatones"]
    .mean()
)

agg.rename(columns={"peatones": "peatones_media"}, inplace=True)

# ---------------------------------------------------------
# NORMALIZACIÓN, ALTURAS Y COLORES
# ---------------------------------------------------------
vmin = float(agg["peatones_media"].min())
vmax = float(agg["peatones_media"].max())

def normalize_value(v, vmin, vmax):
    if vmax <= vmin:
        return 0.5
    t = (v - vmin) / (vmax - vmin)
    return max(0.0, min(1.0, float(t)))

# Alturas: normalizamos 0–1, aplicamos una potencia (gamma)
# para enfatizar diferencias y escalamos a una altura máxima.
HEIGHT_MAX = 1500.0   # súbelo/bájalo si quieres más/menos altura
GAMMA = 1.4           # >1 realza las diferencias en la parte alta

agg["norm"] = agg["peatones_media"].apply(
    lambda v: normalize_value(v, vmin, vmax)
)
agg["height"] = (agg["norm"] ** GAMMA) * HEIGHT_MAX

def lerp(a, b, t):
    return a + (b - a) * t

def gradient_color(t):
    """
    Gradiente:
      0.00 -> azul oscuro      [  5,  20,  70]
      0.25 -> turquesa         [  0, 150, 200]
      0.50 -> verde intenso    [ 80, 220, 120]
      0.75 -> amarillo         [255, 230,  80]
      1.00 -> naranja-rojo     [255, 100,  40]
    """
    t = max(0.0, min(1.0, float(t)))

    # Tramos
    if t <= 0.25:
        t0, c0 = 0.0,  (5, 20, 70)
        t1, c1 = 0.25, (0, 150, 200)
        tt = (t - t0) / (t1 - t0)
    elif t <= 0.5:
        t0, c0 = 0.25, (0, 150, 200)
        t1, c1 = 0.5,  (80, 220, 120)
        tt = (t - t0) / (t1 - t0)
    elif t <= 0.75:
        t0, c0 = 0.5,  (80, 220, 120)
        t1, c1 = 0.75, (255, 230, 80)
        tt = (t - t0) / (t1 - t0)
    else:
        t0, c0 = 0.75, (255, 230, 80)
        t1, c1 = 1.0,  (255, 100, 40)
        tt = (t - t0) / (t1 - t0)

    r = lerp(c0[0], c1[0], tt)
    g = lerp(c0[1], c1[1], tt)
    b = lerp(c0[2], c1[2], tt)
    return [int(r), int(g), int(b), 220]  # alpha 220 para algo de transparencia

# Aplicamos colores
agg["color"] = agg["norm"].apply(gradient_color)

# ---------------------------------------------------------
# MAPA 3D (SIN MAPBOX, ESCENA OSCURA)
# ---------------------------------------------------------
mid_lat = float(agg["lat"].mean())
mid_lon = float(agg["lon"].mean())

layer = pdk.Layer(
    "ColumnLayer",
    data=agg,
    get_position="[lon, lat]",
    get_elevation="height",
    elevation_scale=1,        # ya hemos escalado nosotros
    radius=40,                # radio de cada columna
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=mid_lat,
    longitude=mid_lon,
    zoom=11,
    pitch=55,   # inclinación para vista más “cinemática”
    bearing=15, # un poco de giro
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style=None,  # sin mapa base
    tooltip={
        "text": "Distrito: {distrito}\nMedia: {peatones_media} peatones"
    },
)

st.subheader("Mapa 3D interactivo")
st.pydeck_chart(deck)

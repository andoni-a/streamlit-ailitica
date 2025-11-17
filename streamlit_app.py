import math

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="Mapa 3D de peatones",
    layout="wide",
)

st.title("Mapa 3D de peatones por zona")
st.markdown(
    """
    Visualización 3D de la media de peatones por zona (datos sintéticos).
    Usa el *slider* para elegir la **franja horaria** del día.
    """
)

# ---------- CARGA DE DATOS ----------

@st.cache_data
def load_data():
    # NUEVO: usamos el CSV sintético
    df = pd.read_csv("df_synthetic_pedestrians.csv")

    # Convertimos "hora" (ej. "10:00") a entero 0–23
    def parse_hour(s):
        try:
            return int(str(s).split(":")[0])
        except Exception:
            return None

    df["hour"] = df["hora"].apply(parse_hour)

    # Renombramos por comodidad
    df.rename(
        columns={
            "latitude": "lat",
            "longitude": "lon",
            "noisy_peatones_correlated": "peatones",
        },
        inplace=True,
    )

    # Quitamos filas sin hora o sin coordenadas
    df = df.dropna(subset=["lat", "lon", "hour"])

    return df


df = load_data()

# ---------- SLIDER DE FRANJA HORARIA ----------

st.subheader("Filtro temporal")

min_hour = int(df["hour"].min())
max_hour = int(df["hour"].max())

start_hour, end_hour = st.slider(
    "Selecciona la franja horaria (horas del día)",
    min_value=min_hour,
    max_value=max_hour,
    value=(8, 20),
    step=1,
)

st.markdown(
    f"Mostrando la **media de peatones** entre las "
    f"**{start_hour}:00** y las **{end_hour}:00**."
)

mask = (df["hour"] >= start_hour) & (df["hour"] <= end_hour)
df_filtered = df[mask].copy()

if df_filtered.empty:
    st.warning("No hay datos para esa franja horaria.")
    st.stop()

# ---------- AGREGACIÓN POR ZONA (DISTRITO + LAT/LON) ----------

agg = (
    df_filtered.groupby(
        ["distrito", "lat", "lon"], as_index=False
    )["peatones"]
    .mean()
)

agg.rename(columns={"peatones": "peatones_media"}, inplace=True)

# ---------- ESCALA DE COLORES (AZUL → VERDE → AMARILLO → ROJO) ----------

vmin = agg["peatones_media"].min()
vmax = agg["peatones_media"].max()

def value_to_color(v, vmin, vmax):
    """Devuelve [R, G, B, A] con gradiente azul->verde->amarillo->rojo."""
    if math.isnan(v):
        return [200, 200, 200, 80]

    if vmax > vmin:
        t = (v - vmin) / (vmax - vmin)
    else:
        t = 0.5

    # 0 -> azul oscuro   (0, 0, 130)
    # 0.33 -> verde      (0, 255, 0)
    # 0.66 -> amarillo   (255, 255, 0)
    # 1 -> rojo          (255, 0, 0)

    if t < 0.33:
        # azul -> verde
        tt = t / 0.33
        r = 0
        g = 0 + (255 - 0) * tt
        b = 130 + (0 - 130) * tt
    elif t < 0.66:
        # verde -> amarillo
        tt = (t - 0.33) / 0.33
        r = 0 + (255 - 0) * tt
        g = 255
        b = 0
    else:
        # amarillo -> rojo
        tt = (t - 0.66) / 0.34
        r = 255
        g = 255 + (0 - 255) * tt
        b = 0

    return [int(r), int(g), int(b), 200]


agg["color"] = agg["peatones_media"].apply(
    lambda v: value_to_color(v, vmin, vmax)
)

# ---------- MAPA 3D (SIN MAPBOX, FONDO OSCURO/NEUTRO) ----------

mid_lat = agg["lat"].mean()
mid_lon = agg["lon"].mean()

layer = pdk.Layer(
    "ColumnLayer",
    data=agg,
    get_position="[lon, lat]",
    get_elevation="peatones_media",
    elevation_scale=0.5,   # ajusta si quieres más/menos altura
    radius=40,             # tamaño de las columnas
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=mid_lat,
    longitude=mid_lon,
    zoom=11,
    pitch=50,
    bearing=0,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style=None,  # sin mapa base (solo tus datos)
    tooltip={
        "text": "Distrito: {distrito}\nMedia: {peatones_media} peatones"
    },
)

st.subheader("Mapa 3D interactivo")
st.pydeck_chart(deck)

# ---------- TABLA RESUMEN (OPCIONAL) ----------

with st.expander("Ver tabla de medias por zona"):
    st.dataframe(
        agg[["distrito", "peatones_media", "lat", "lon"]]
        .sort_values("peatones_media", ascending=False)
        .reset_index(drop=True)
    )

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
    Visualización 3D de la media de peatones por zona en Madrid.
    Usa el *slider* inferior para elegir la **franja horaria** del día.
    """
)

# ---------- PASO 1: CARGA Y LIMPIEZA DE DATOS ----------

@st.cache_data
def load_data():
    # Ajusta el nombre del fichero si hace falta
    df = pd.read_csv("PEATONES_2024.csv", sep=";")

    # Hora -> entero [0..23]
    def parse_hour(s):
        try:
            return int(str(s).split(":")[0])
        except Exception:
            return None

    df["hour"] = df["hora"].apply(parse_hour)

    # Coordenadas vienen como "40.417.386" -> 40.417386
    def fix_coord(s):
        s = str(s).strip()
        if s == "" or s.lower() == "nan":
            return np.nan

        neg = s.startswith("-")
        if neg:
            s2 = s[1:]
        else:
            s2 = s

        digits = s2.replace(".", "")
        if not digits.isdigit():
            return np.nan

        val = int(digits) / 1e6
        return -val if neg else val

    df["lat"] = df["latitude"].apply(fix_coord)
    df["lon"] = df["longitude"].apply(fix_coord)

    # Quitamos filas sin coordenadas u hora
    df = df.dropna(subset=["lat", "lon", "hour"])

    return df


df = load_data()

# ---------- PASO 2: SLIDER DE FRANJA HORARIA ----------

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

# ---------- PASO 3: AGREGACIÓN POR ZONA ----------

# Considero cada device_id/dirección como una "zona"
agg = (
    df_filtered.groupby(
        ["device_id", "direccion", "lat", "lon"], as_index=False
    )["peatones"]
    .mean()
)

agg.rename(columns={"peatones": "peatones_media"}, inplace=True)

# ---------- PASO 4: ESCALA DE COLORES (AZUL -> VERDE -> AMARILLO -> ROJO) ----------

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

# ---------- PASO 5: MAPA 3D OSCURO SIN MAPBOX ----------

mid_lat = agg["lat"].mean()
mid_lon = agg["lon"].mean()

layer = pdk.Layer(
    "ColumnLayer",
    data=agg,
    get_position="[lon, lat]",
    get_elevation="peatones_media",
    elevation_scale=0.5,  # ajusta para más/menos altura
    radius=60,            # radio (en metros aprox.)
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
    map_style=None,                 # <<< SIN MAPBOX
    background_color=[0, 0, 0, 255],  # fondo negro
    tooltip={
        "text": "Zona: {direccion}\nMedia: {peatones_media} peatones"
    },
)

st.subheader("Mapa 3D interactivo")
st.pydeck_chart(deck)


# ---------- PASO 6: TABLA RESUMEN (opcional) ----------

with st.expander("Ver tabla de medias por zona"):
    st.dataframe(
        agg[["direccion", "peatones_media", "lat", "lon"]]
        .sort_values("peatones_media", ascending=False)
        .reset_index(drop=True)
    )

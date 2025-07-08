import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from io import BytesIO
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter, sobel, label
import tempfile
import pandas as pd
import os
import time
import subprocess
import richdem as rd

st.set_page_config(layout="wide")
st.title("🌕 Lunar Hazard Detection: Landslides + Boulders")

# Sidebar status box
st.sidebar.markdown("### 📂 Processing Summary")
status_lines = []
if dtm_file := st.sidebar.file_uploader("Upload DTM (Digital Terrain Model) GeoTIFF", type=["tif"]):
    status_lines.append("✅ DTM File Loaded")
else:
    status_lines.append("📂 Awaiting DTM File")
if ohrc_file := st.sidebar.file_uploader("Upload OHRC (High-Res Optical) File", type=["tif", "img", "IMG"]):
    status_lines.append("✅ OHRC File Loaded")
else:
    status_lines.append("📂 Awaiting OHRC File")
for line in status_lines:
    st.sidebar.markdown(f"- {line}")

# Tabs UI Layout
st.markdown("---")
st.info("ℹ️ Upload DTM for terrain analysis and OHRC for boulder detection. Downloadable results will appear below.")
tabs = st.tabs(["🪨 Landslide Detection", "🧱 Boulder Detection", "📘 About the App"])

# -------- DTM PROCESSING --------
with tabs[0]:
    if dtm_file is not None:
        min_region_pixels = st.slider("Minimum region pixel size", 100, 5000, 1000, step=100)
        show_sources = st.checkbox("Show Source Points", value=True)
        show_landslides = st.checkbox("Show Landslide Overlay", value=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmpfile:
            tmpfile.write(dtm_file.read())
            tmp_path = tmpfile.name

        st.success("✅ DTM File loaded successfully")

        try:
            with rasterio.open(tmp_path) as src:
                transform = src.transform
                pixel_size_x = transform[0]
                pixel_size_y = -transform[4]
                moon_radius = 1737400
                deg_to_rad = np.pi / 180
                meters_per_degree = moon_radius * deg_to_rad
                pixel_size_x_m = pixel_size_x * meters_per_degree
                pixel_size_y_m = pixel_size_y * meters_per_degree
                pixel_area_m2 = pixel_size_x_m * pixel_size_y_m
                st.markdown(f"#### DTM Pixel size: {pixel_size_x_m:.2f} m x {pixel_size_y_m:.2f} m")

                dtm = src.read(1, out_dtype='float32')

                slope_threshold = 30
                roughness_kernel = 9
                tile_size = 512
                height, width = dtm.shape

                landslide_mask = np.zeros_like(dtm, dtype=bool)
                labeled_mask = np.zeros_like(dtm, dtype=int)
                label_offset = 1

                source_coords = []

                for row in range(0, height, tile_size):
                    for col in range(0, width, tile_size):
                        with st.spinner(f"🔄 Processing tile: row={row}, col={col}"):
                            chunk = dtm[row:row+tile_size, col:col+tile_size]
                            if np.std(chunk) < 1:
                                continue

                            dy, dx = np.gradient(chunk, pixel_size_y_m, pixel_size_x_m)
                            slope_chunk = gaussian_filter(np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))), sigma=1)
                            smooth_chunk = gaussian_filter(chunk, sigma=roughness_kernel / 6)
                            roughness_chunk = gaussian_filter(np.abs(chunk - smooth_chunk), sigma=1)
                            rough_thresh = np.percentile(roughness_chunk, 95)

                            mask = (slope_chunk > slope_threshold) & (roughness_chunk > rough_thresh)
                            landslide_mask[row:row+chunk.shape[0], col:col+chunk.shape[1]] |= mask

                            chunk_rd = rd.rdarray(chunk, no_data=-9999)
                            chunk_rd.geotransform = [0, 1, 0, 0, 0, -1]
                            flow_acc = rd.FlowAccumulation(chunk_rd, method='D8')

                            labeled, nfeat = label(mask)
                            for i in range(1, nfeat + 1):
                                region = (labeled == i)
                                if np.sum(region) < min_region_pixels:
                                    labeled[region] = 0
                                else:
                                    ys, xs = np.where(region)
                                    if len(ys) > 0 and len(xs) > 0:
                                        scores = slope_chunk[ys, xs] * flow_acc[ys, xs]
                                        best_idx = np.argmax(scores)
                                        source_row = row + ys[best_idx]
                                        source_col = col + xs[best_idx]
                                        x_geo, y_geo = transform * (source_col, source_row)
                                        source_coords.append((x_geo, y_geo))

                            labeled_mask[row:row+chunk.shape[0], col:col+chunk.shape[1]] += (labeled + label_offset)
                            label_offset += nfeat

                total_pixels = np.sum(landslide_mask)
                total_regions = labeled_mask.max()
                total_sources = len(source_coords)
                st.metric("🪨 Landslide Pixels", total_pixels)
                st.metric("🖼 Landslide Regions", total_regions)
                st.metric("⭐ Source Points", total_sources)

                with st.expander("🌋 Side-by-Side Viewer (DTM vs Landslide Output)", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Input DTM")
                        fig1, ax1 = plt.subplots()
                        ax1.imshow(dtm, cmap='gray')
                        ax1.set_title("Original DTM")
                        ax1.axis('off')
                        st.pyplot(fig1)
                    with col2:
                        st.markdown("#### Detected Landslides")
                        fig2, ax2 = plt.subplots(figsize=(8, 8))
                        ax2.imshow(dtm, cmap='gray')
                        if show_landslides:
                            ax2.imshow(np.ma.masked_where(~landslide_mask, landslide_mask), cmap='hot', alpha=0.6)
                        if show_sources and source_coords:
                            xs, ys = zip(*[~transform * (x, y) for x, y in source_coords])
                            ax2.scatter(xs, ys, c='cyan', s=25, edgecolors='black', label="Source Points", alpha=0.9, marker='*')
                            ax2.legend(loc='lower left', fontsize='small', frameon=True)
                        ax2.set_title("Landslide Zones + Sources (Overlay)", fontsize=14, fontweight='bold')
                        ax2.axis('off')
                        plt.tight_layout()
                        st.pyplot(fig2)

                fig_buf = BytesIO()
                fig2.savefig(fig_buf, format='png', dpi=300)
                fig_buf.seek(0)
                st.download_button("🗕️ Download Landslide Map Image", fig_buf, file_name="landslide_map_with_sources.png")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as out_raster:
                    with rasterio.open(out_raster.name, 'w', driver='GTiff',
                                       height=labeled_mask.shape[0], width=labeled_mask.shape[1],
                                       count=1, dtype='int32', crs=src.crs,
                                       transform=transform) as dst:
                        dst.write(labeled_mask.astype('int32'), 1)

                    with open(out_raster.name, "rb") as f:
                        st.download_button("🗕️ Download Landslide GeoTIFF", f, file_name="landslide_labeled_map.tif")

                if source_coords:
                    source_df = pd.DataFrame(source_coords, columns=["Longitude", "Latitude"])
                    csv_buf = BytesIO()
                    source_df.to_csv(csv_buf, index=False)
                    csv_buf.seek(0)
                    st.download_button("🗕️ Download Landslide Sources CSV", csv_buf, file_name="landslide_sources.csv")

        except Exception as e:
            st.error(f"❌ Failed to open DTM TIFF file: {e}")

# -------- BOULDER PROCESSING --------
with tabs[1]:
    if ohrc_file is not None:
        blur_sigma = st.slider("Gaussian Blur Sigma", 0.5, 5.0, 2.0, step=0.1)
        intensity_percentile = st.slider("Detection Threshold (Percentile)", 95.0, 99.9, 99.5, step=0.1)

        file_ext = os.path.splitext(ohrc_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmpfile:
            tmpfile.write(ohrc_file.read())
            raw_ohrc_path = tmpfile.name

        st.success("✅ OHRC File loaded successfully")

        try:
            if file_ext == ".img":
                converted_path = raw_ohrc_path + ".tif"
                cmd = ["gdal_translate", raw_ohrc_path, converted_path]
                subprocess.run(cmd, check=True)
                ohrc_path = converted_path
            else:
                ohrc_path = raw_ohrc_path

            with rasterio.open(ohrc_path) as src:
                image = src.read(1, out_dtype='float32')
                blurred = gaussian_filter(image, sigma=blur_sigma)
                high_pass = image - blurred
                boulder_candidates = high_pass > np.percentile(high_pass, intensity_percentile)
                labeled_boulders, num_boulders = label(boulder_candidates)

                st.metric("🧱 Boulders Detected", num_boulders)

                with st.expander("🧱 Side-by-Side Viewer (OHRC vs Boulder Output)", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Input OHRC Image")
                        fig3, ax3 = plt.subplots()
                        ax3.imshow(image, cmap='gray')
                        ax3.set_title("Original OHRC")
                        ax3.axis('off')
                        st.pyplot(fig3)
                    with col2:
                        st.markdown("#### Detected Boulders")
                        fig4, ax4 = plt.subplots()
                        ax4.imshow(image, cmap='gray')
                        ax4.imshow(labeled_boulders, cmap='Reds', alpha=0.4)
                        ax4.set_title("Detected Boulders")
                        ax4.axis('off')
                        st.pyplot(fig4)

                fig_buf_b = BytesIO()
                fig4.savefig(fig_buf_b, format='png', dpi=300)
                fig_buf_b.seek(0)
                st.download_button("🧱 Download Boulder Map Image", fig_buf_b, file_name="boulder_map_overlay.png")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as out_raster:
                    with rasterio.open(out_raster.name, 'w', driver='GTiff',
                                       height=labeled_boulders.shape[0], width=labeled_boulders.shape[1],
                                       count=1, dtype='int32', crs=src.crs,
                                       transform=src.transform) as dst:
                        dst.write(labeled_boulders.astype('int32'), 1)

                    with open(out_raster.name, "rb") as f:
                        st.download_button("🧱 Download Boulder GeoTIFF", f, file_name="boulder_labeled_map.tif")

                boulder_coords = []
                for region_label in range(1, num_boulders + 1):
                    mask = labeled_boulders == region_label
                    if np.any(mask):
                        rows, cols = np.where(mask)
                        row_mean, col_mean = np.mean(rows), np.mean(cols)
                        x_geo, y_geo = src.transform * (col_mean, row_mean)
                        size = np.sqrt(np.sum(mask))
                        boulder_coords.append((x_geo, y_geo, size))

                if boulder_coords:
                    boulder_df = pd.DataFrame(boulder_coords, columns=["Longitude", "Latitude", "Approx_Size"])
                    csv_buf_b = BytesIO()
                    boulder_df.to_csv(csv_buf_b, index=False)
                    csv_buf_b.seek(0)
                    st.download_button("🧱 Download Boulder Details CSV", csv_buf_b, file_name="boulder_locations.csv")

        except Exception as e:
            st.error(f"❌ Failed to process OHRC file: {e}")

# Tab 3: About Section
with tabs[2]:
    st.markdown("### 📘 About the App")
    st.markdown("""
    This web application is developed for lunar hazard detection using Digital Terrain Models (DTM) and Optical High-Resolution Camera (OHRC) imagery.

    - **Landslide Detection** uses slope and roughness metrics.
    - **Boulder Detection** applies high-pass filtering on OHRC data.

    #### 📁 Supported Formats
    - DTM: GeoTIFF (.tif)
    - OHRC: .tif, .img, .IMG

    #### 👨‍💻 Developer
    - Developed as part of a 2025 lunar research project on terrain hazard mapping.
    """)

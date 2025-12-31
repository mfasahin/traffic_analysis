"""Streamlit web application for traffic analysis."""

import streamlit as st
import cv2
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Tuple
import time
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# Project imports
from domain.entities.roi import ROI
from domain.entities.multi_roi import MultiROI, LaneROI
from domain.entities.speed_calibration import SpeedCalibration
from infrastructure.detection.yolo_vehicle_detector import YOLOVehicleDetector
from infrastructure.video.opencv_video_processor import OpenCVVideoProcessor
from application.services.traffic_analyzer import TrafficAnalyzerService
from presentation.visualization.video_visualizer import VideoVisualizer
from presentation.reporting.statistics_reporter import StatisticsReporter


# Page configuration
st.set_page_config(
    page_title="Traffic Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .progress-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .video-container {
        background: #000;
        border-radius: 1rem;
        padding: 0.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stats-overlay {
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'video_file' not in st.session_state:
        st.session_state.video_file = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'roi_points' not in st.session_state:
        st.session_state.roi_points = []
    if 'multi_roi' not in st.session_state:
        st.session_state.multi_roi = None
    if 'calibration' not in st.session_state:
        st.session_state.calibration = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'output_video_path' not in st.session_state:
        st.session_state.output_video_path = None
    if 'frame_preview' not in st.session_state:
        st.session_state.frame_preview = None
    if 'roi_click_points' not in st.session_state:
        st.session_state.roi_click_points = []
    if 'current_lane_points' not in st.session_state:
        st.session_state.current_lane_points = []
    if 'calibration_points' not in st.session_state:
        st.session_state.calibration_points = []
    if 'original_image_size' not in st.session_state:
        st.session_state.original_image_size = None


def load_video_preview(video_path: str) -> Optional[np.ndarray]:
    """Load first frame of video for preview."""
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Store original image size
            st.session_state.original_image_size = (frame_rgb.shape[1], frame_rgb.shape[0])  # (width, height)
            cap.release()
            return frame_rgb
    cap.release()
    return None


def draw_roi_on_image(image: np.ndarray, roi_points: List[Tuple[int, int]], color=(255, 0, 0)) -> np.ndarray:
    """Draw ROI polygon on image (RGB format)."""
    if len(roi_points) < 2:
        return image
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_copy = img_bgr.copy()
    points = np.array(roi_points, dtype=np.int32)
    
    # Convert color to BGR
    color_bgr = (color[2], color[1], color[0])
    
    # Draw filled polygon with transparency
    overlay = img_copy.copy()
    cv2.fillPoly(overlay, [points], color_bgr)
    cv2.addWeighted(overlay, 0.3, img_copy, 0.7, 0, img_copy)
    
    # Draw outline
    cv2.polylines(img_copy, [points], isClosed=True, color=color_bgr, thickness=2)
    
    # Draw points
    for i, point in enumerate(roi_points):
        cv2.circle(img_copy, point, 5, color_bgr, -1)
        cv2.putText(img_copy, str(i+1), (point[0]+10, point[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
    
    # Convert back to RGB
    return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🚗 Traffic Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Adımlar")
        st.markdown("""
        1. **Video Yükle** - Analiz edilecek video
        2. **ROI Seç** - Yol alanını tanımla (opsiyonel)
        3. **Kalibrasyon** - Hız için kalibrasyon (opsiyonel)
        4. **Analiz Ayarları** - Parametreleri ayarla
        5. **Analiz Çalıştır** - İşlemi başlat
        6. **Sonuçları Görüntüle** - İstatistikler ve video
        """)
        
        st.divider()
        
        st.header("⚙️ Ayarlar")
        enable_tracking = st.checkbox("Araç Takibi", value=True, help="Hız ve yön analizi için gerekli")
        enable_smoothing = st.checkbox("Temporal Smoothing", value=True, help="Occlusion olaylarını filtreler")
        
        if enable_tracking:
            speed_limit = st.slider("Hız Limiti (km/h)", 30, 120, 50)
        else:
            speed_limit = 50
        
        confidence = st.slider("Güven Eşiği", 0.1, 0.9, 0.25, 0.05)
        skip_frames = st.slider("Frame Atlama", 0, 30, 0, help="0 = tüm frame'ler, 10 = her 11. frame")
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📹 Video Yükleme", 
        "🎯 ROI Seçimi", 
        "📏 Kalibrasyon", 
        "🚀 Analiz", 
        "📊 Sonuçlar"
    ])
    
    # Tab 1: Video Upload
    with tab1:
        st.header("Video Yükleme")
        
        uploaded_file = st.file_uploader(
            "Video dosyası seçin (MP4, AVI, MOV)",
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
                st.session_state.video_file = uploaded_file.name
            
            st.success(f"✅ Video yüklendi: {uploaded_file.name}")
            
            # Preview first frame
            preview = load_video_preview(st.session_state.video_path)
            if preview is not None:
                st.session_state.frame_preview = preview
                st.image(preview, caption="Video Önizleme (İlk Frame)", use_container_width=True)
                
                # Video info
                cap = cv2.VideoCapture(st.session_state.video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Genişlik", f"{width}px")
                with col2:
                    st.metric("Yükseklik", f"{height}px")
                with col3:
                    st.metric("FPS", f"{fps:.2f}")
                with col4:
                    st.metric("Süre", f"{duration:.1f}s")
        else:
            st.info("👆 Lütfen bir video dosyası yükleyin")
    
    # Tab 2: ROI Selection
    with tab2:
        st.header("ROI (Region of Interest) Seçimi")
        st.markdown("Yol alanını tanımlamak için ROI seçebilirsiniz. Bu, yoğunluk hesaplamasını daha doğru yapar.")
        
        if st.session_state.video_path is None:
            st.warning("⚠️ Önce video yükleyin!")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("ROI Seçimi")
                use_roi = st.checkbox("ROI kullan", value=False)
                
                if use_roi:
                    st.markdown("""
                    **Nasıl kullanılır:**
                    1. Aşağıdaki koordinatları manuel olarak girin
                    2. Veya ROI seçiciyi kullanın (geliştiriliyor)
                    """)
                    
                    roi_method = st.radio(
                        "ROI Seçim Yöntemi",
                        ["Resim Üzerinden Tıklayarak", "Manuel Koordinatlar", "Çoklu ROI (Şeritler)"],
                        horizontal=False
                    )
                    
                    if roi_method == "Resim Üzerinden Tıklayarak":
                        st.markdown("""
                        **Nasıl kullanılır:**
                        1. Aşağıdaki resim üzerinde yol alanını tanımlamak için noktalara tıklayın
                        2. En az 3 nokta seçin
                        3. "ROI'yi Kaydet" butonuna tıklayın
                        """)
                        
                        if st.session_state.frame_preview is not None:
                            # Display image with click coordinates
                            st.markdown("**Resim üzerine tıklayarak nokta ekleyin:**")
                            
                            col_btn1, col_btn2, col_btn3 = st.columns(3)
                            with col_btn1:
                                if st.button("🔄 Sıfırla", use_container_width=True):
                                    st.session_state.roi_click_points = []
                                    st.rerun()
                            with col_btn2:
                                if st.button("↩️ Son Noktayı Sil", use_container_width=True):
                                    if st.session_state.roi_click_points:
                                        st.session_state.roi_click_points.pop()
                                    st.rerun()
                            with col_btn3:
                                if st.button("💾 ROI'yi Kaydet", use_container_width=True, type="primary"):
                                    if len(st.session_state.roi_click_points) >= 3:
                                        try:
                                            roi = ROI(st.session_state.roi_click_points)
                                            st.session_state.roi_points = st.session_state.roi_click_points
                                            st.session_state.multi_roi = None
                                            st.success(f"✅ ROI kaydedildi! Alan: {roi.get_area()} piksel")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ ROI hatası: {e}")
                                    else:
                                        st.error("❌ En az 3 nokta seçmelisiniz!")
                            
                            # Draw current points on image
                            img_with_points = st.session_state.frame_preview.copy()
                            if st.session_state.roi_click_points:
                                img_with_points = draw_roi_on_image(
                                    img_with_points,
                                    st.session_state.roi_click_points
                                )
                            
                            # Use streamlit-image-coordinates for click detection
                            try:
                                # Get display size (width parameter)
                                display_width = 700
                                clicked_point = streamlit_image_coordinates(
                                    img_with_points,
                                    key="roi_selector",
                                    width=display_width
                                )
                                
                                if clicked_point is not None and st.session_state.original_image_size:
                                    # Get clicked coordinates (in display size)
                                    display_x = clicked_point['x']
                                    display_y = clicked_point['y']
                                    
                                    # Get original and display dimensions
                                    orig_width, orig_height = st.session_state.original_image_size
                                    display_height = int(orig_height * (display_width / orig_width))
                                    
                                    # Scale coordinates to original image size
                                    scale_x = orig_width / display_width
                                    scale_y = orig_height / display_height
                                    
                                    x = int(display_x * scale_x)
                                    y = int(display_y * scale_y)
                                    
                                    # Clamp to image bounds
                                    x = max(0, min(x, orig_width - 1))
                                    y = max(0, min(y, orig_height - 1))
                                    
                                    # Add point if not already exists (with small tolerance)
                                    point_exists = False
                                    for existing_point in st.session_state.roi_click_points:
                                        if abs(existing_point[0] - x) < 10 and abs(existing_point[1] - y) < 10:
                                            point_exists = True
                                            break
                                    
                                    if not point_exists:
                                        st.session_state.roi_click_points.append((x, y))
                                        st.rerun()
                            except Exception as e:
                                # Fallback to manual input if library not available
                                st.warning(f"Tıklama özelliği yüklenemedi: {e}")
                                st.info("Manuel koordinat girişi kullanın veya 'pip install streamlit-image-coordinates' komutunu çalıştırın")
                            
                            st.write(f"**Seçilen Noktalar:** {len(st.session_state.roi_click_points)}/3 (minimum)")
                            if st.session_state.roi_click_points:
                                for i, point in enumerate(st.session_state.roi_click_points):
                                    st.write(f"  Nokta {i+1}: ({point[0]}, {point[1]})")
                            
                            # Show preview with current points - always show the image with points
                            st.image(img_with_points, caption="ROI Önizleme - Noktalara tıklayarak ekleyin", use_container_width=True)
                        else:
                            st.warning("⚠️ Video önizlemesi yüklenemedi")
                    
                    elif roi_method == "Manuel Koordinatlar":
                        st.markdown("**Koordinat Formatı:** `x1,y1,x2,y2,x3,y3,...` (en az 3 nokta)")
                        roi_coords_str = st.text_input(
                            "ROI Koordinatları",
                            placeholder="Örn: 100,200,800,200,900,600,50,600",
                            help="Virgülle ayrılmış koordinatlar"
                        )
                        
                        if roi_coords_str:
                            try:
                                coords = [int(x.strip()) for x in roi_coords_str.split(',')]
                                if len(coords) >= 6 and len(coords) % 2 == 0:
                                    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                                    st.session_state.roi_points = points
                                    
                                    # Draw ROI on preview
                                    if st.session_state.frame_preview is not None:
                                        img_with_roi = draw_roi_on_image(
                                            st.session_state.frame_preview.copy(),
                                            points
                                        )
                                        st.image(img_with_roi, caption="ROI Önizleme", use_container_width=True)
                                        
                                        try:
                                            roi = ROI(points)
                                            st.success(f"✅ ROI oluşturuldu! Alan: {roi.get_area()} piksel")
                                            st.session_state.multi_roi = None
                                        except Exception as e:
                                            st.error(f"❌ ROI hatası: {e}")
                                else:
                                    st.error("❌ En az 3 nokta (6 koordinat) gerekli ve çift sayıda koordinat olmalı")
                            except ValueError:
                                st.error("❌ Geçersiz koordinat formatı")
                    
                    else:  # Multi-ROI
                        st.markdown("**Çoklu Şerit için:** Her şerit için ayrı ROI tanımlayın")
                        num_lanes = st.number_input("Şerit Sayısı", min_value=1, max_value=10, value=2)
                        
                        lanes = []
                        for i in range(num_lanes):
                            st.markdown(f"### Şerit {i+1}")
                            lane_id = st.text_input(f"Şerit ID", value=f"lane_{i+1}", key=f"lane_id_{i}")
                            lane_name = st.text_input(f"Şerit Adı (opsiyonel)", key=f"lane_name_{i}")
                            lane_coords = st.text_input(
                                f"Koordinatlar (x1,y1,x2,y2,...)",
                                key=f"lane_coords_{i}",
                                placeholder="100,200,800,200,900,600,50,600"
                            )
                            
                            if lane_coords:
                                try:
                                    coords = [int(x.strip()) for x in lane_coords.split(',')]
                                    if len(coords) >= 6 and len(coords) % 2 == 0:
                                        points = [(coords[j], coords[j+1]) for j in range(0, len(coords), 2)]
                                        try:
                                            roi = ROI(points)
                                            lane = LaneROI(
                                                roi=roi,
                                                lane_id=lane_id,
                                                lane_name=lane_name if lane_name else None
                                            )
                                            lanes.append(lane)
                                            st.success(f"✅ Şerit {i+1} eklendi")
                                        except Exception as e:
                                            st.error(f"❌ Şerit {i+1} hatası: {e}")
                                except ValueError:
                                    st.error(f"❌ Geçersiz koordinat formatı (Şerit {i+1})")
                        
                        if lanes:
                            try:
                                multi_roi = MultiROI(lanes=lanes)
                                st.session_state.multi_roi = multi_roi
                                st.session_state.roi_points = []
                                st.success(f"✅ {len(lanes)} şeritli Multi-ROI oluşturuldu!")
                            except Exception as e:
                                st.error(f"❌ Multi-ROI hatası: {e}")
                else:
                    st.info("ROI kullanılmıyor - Tüm frame alanı kullanılacak")
                    st.session_state.roi_points = []
                    st.session_state.multi_roi = None
            
            with col2:
                st.subheader("ROI Bilgileri")
                if st.session_state.multi_roi:
                    st.write(f"**Şerit Sayısı:** {len(st.session_state.multi_roi.lanes)}")
                    st.write(f"**Toplam Alan:** {st.session_state.multi_roi.get_total_area()} piksel")
                    for lane in st.session_state.multi_roi.lanes:
                        st.write(f"- {lane.lane_id}: {lane.roi.get_area()} piksel")
                elif st.session_state.roi_points:
                    try:
                        roi = ROI(st.session_state.roi_points)
                        st.write(f"**Alan:** {roi.get_area()} piksel")
                        st.write(f"**Nokta Sayısı:** {len(st.session_state.roi_points)}")
                    except:
                        pass
    
    # Tab 3: Calibration
    with tab3:
        st.header("Hız Kalibrasyonu")
        st.markdown("Hız hesaplaması için kalibrasyon yapabilirsiniz. Kalibrasyon olmadan hız piksel/saniye cinsinden gösterilir.")
        
        if st.session_state.video_path is None:
            st.warning("⚠️ Önce video yükleyin!")
        else:
            calibration_method = st.radio(
                "Kalibrasyon Yöntemi",
                ["Kalibrasyon Yok", "Piksel/Metre Oranı", "Referans Mesafe", "ROI ve Şerit Bilgisi"],
                horizontal=False
            )
            
            if calibration_method == "Kalibrasyon Yok":
                st.info("ℹ️ Kalibrasyon yapılmadan hız piksel/saniye cinsinden hesaplanacak")
                st.session_state.calibration = None
            
            elif calibration_method == "Piksel/Metre Oranı":
                pixels_per_meter = st.number_input(
                    "Piksel/Metre Oranı",
                    min_value=0.1,
                    max_value=1000.0,
                    value=28.57,
                    step=0.1,
                    help="1 metre = kaç piksel?"
                )
                
                if st.button("Kalibrasyon Oluştur"):
                    cap = cv2.VideoCapture(st.session_state.video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    st.session_state.calibration = SpeedCalibration(
                        pixels_per_meter=pixels_per_meter,
                        fps=fps
                    )
                    st.success(f"✅ Kalibrasyon oluşturuldu: {pixels_per_meter} px/m")
            
            elif calibration_method == "Referans Mesafe":
                st.markdown("""
                **Referans Mesafe ile Kalibrasyon:**
                Video'da bilinen bir mesafeyi ölçün (örn: şerit genişliği 3.5m)
                """)
                
                measure_method = st.radio(
                    "Ölçüm Yöntemi",
                    ["Resim Üzerinden İki Nokta", "Manuel Giriş"],
                    horizontal=True
                )
                
                if measure_method == "Resim Üzerinden İki Nokta":
                    if st.session_state.frame_preview is not None:
                        st.markdown("**Resim üzerinde iki noktaya tıklayarak mesafeyi ölçün:**")
                        
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("🔄 Sıfırla", key="reset_calib", use_container_width=True):
                                st.session_state.calibration_points = []
                                st.rerun()
                        
                        # Draw calibration line
                        img_calib = st.session_state.frame_preview.copy()
                        if len(st.session_state.calibration_points) >= 1:
                            # Draw first point
                            pt1 = st.session_state.calibration_points[0]
                            cv2.circle(img_calib, pt1, 8, (0, 255, 0), -1)
                            cv2.putText(img_calib, "1", (pt1[0]+10, pt1[1]-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            if len(st.session_state.calibration_points) >= 2:
                                # Draw second point and line
                                pt2 = st.session_state.calibration_points[1]
                                cv2.circle(img_calib, pt2, 8, (0, 255, 0), -1)
                                cv2.putText(img_calib, "2", (pt2[0]+10, pt2[1]-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.line(img_calib, pt1, pt2, (0, 255, 0), 2)
                                
                                # Calculate distance
                                dx = pt2[0] - pt1[0]
                                dy = pt2[1] - pt1[1]
                                distance_pixels = (dx**2 + dy**2) ** 0.5
                                
                                # Show distance on image
                                mid_x = (pt1[0] + pt2[0]) // 2
                                mid_y = (pt1[1] + pt2[1]) // 2
                                cv2.putText(img_calib, f"{distance_pixels:.1f} px", 
                                           (mid_x - 50, mid_y - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                
                                st.session_state.calibration_pixels = distance_pixels
                        
                        try:
                            # Get display size (width parameter)
                            display_width = 700
                            clicked_point = streamlit_image_coordinates(
                                img_calib,
                                key="calibration_selector",
                                width=display_width
                            )
                            
                            if clicked_point is not None and len(st.session_state.calibration_points) < 2 and st.session_state.original_image_size:
                                # Get clicked coordinates (in display size)
                                display_x = clicked_point['x']
                                display_y = clicked_point['y']
                                
                                # Get original and display dimensions
                                orig_width, orig_height = st.session_state.original_image_size
                                display_height = int(orig_height * (display_width / orig_width))
                                
                                # Scale coordinates to original image size
                                scale_x = orig_width / display_width
                                scale_y = orig_height / display_height
                                
                                x = int(display_x * scale_x)
                                y = int(display_y * scale_y)
                                
                                # Clamp to image bounds
                                x = max(0, min(x, orig_width - 1))
                                y = max(0, min(y, orig_height - 1))
                                
                                # Check if point already exists
                                point_exists = False
                                for existing_point in st.session_state.calibration_points:
                                    if abs(existing_point[0] - x) < 10 and abs(existing_point[1] - y) < 10:
                                        point_exists = True
                                        break
                                
                                if not point_exists:
                                    st.session_state.calibration_points.append((x, y))
                                    st.rerun()
                        except Exception:
                            st.info("Tıklama özelliği için 'pip install streamlit-image-coordinates' gerekli")
                        
                        st.image(img_calib, caption="Kalibrasyon Ölçümü - İki noktaya tıklayın", use_container_width=True)
                        
                        if len(st.session_state.calibration_points) == 2:
                            st.success(f"✅ Mesafe ölçüldü: {st.session_state.calibration_pixels:.1f} piksel")
                            
                            ref_meters = st.number_input(
                                "Gerçek Mesafe (metre)",
                                min_value=0.1,
                                value=3.5,
                                step=0.1,
                                help="Ölçtüğünüz mesafenin gerçek dünya değeri (örn: şerit genişliği 3.5m)"
                            )
                            
                            if st.button("💾 Kalibrasyon Oluştur", type="primary"):
                                cap = cv2.VideoCapture(st.session_state.video_path)
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                cap.release()
                                
                                st.session_state.calibration = SpeedCalibration.from_reference_distance(
                                    st.session_state.calibration_pixels, ref_meters, fps
                                )
                                calculated_ppm = st.session_state.calibration_pixels / ref_meters
                                st.success(f" Kalibrasyon oluşturuldu: {calculated_ppm:.2f} px/m")
                    else:
                        st.warning(" Video önizlemesi yüklenemedi")
                
                else:  # Manuel giriş
                    col1, col2 = st.columns(2)
                    with col1:
                        ref_pixels = st.number_input("Piksel Cinsinden Mesafe", min_value=1, value=100)
                    with col2:
                        ref_meters = st.number_input("Metre Cinsinden Mesafe", min_value=0.1, value=3.5, step=0.1)
                    
                    if st.button("Kalibrasyon Oluştur"):
                        cap = cv2.VideoCapture(st.session_state.video_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        
                        st.session_state.calibration = SpeedCalibration.from_reference_distance(
                            ref_pixels, ref_meters, fps
                        )
                        calculated_ppm = ref_pixels / ref_meters
                        st.success(f"✅ Kalibrasyon oluşturuldu: {calculated_ppm:.2f} px/m")
            
            elif calibration_method == "ROI ve Şerit Bilgisi":
                if st.session_state.multi_roi is None:
                    st.warning("⚠️ Önce Multi-ROI oluşturun!")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        roi_width = st.number_input("ROI Genişliği (piksel)", min_value=1, value=800)
                    with col2:
                        num_lanes = st.number_input("Şerit Sayısı", min_value=1, max_value=10, value=2)
                    with col3:
                        lane_width = st.number_input("Şerit Genişliği (metre)", min_value=0.1, value=3.5, step=0.1)
                    
                    if st.button("Kalibrasyon Oluştur"):
                        cap = cv2.VideoCapture(st.session_state.video_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        
                        st.session_state.calibration = SpeedCalibration.from_roi_and_lane_width(
                            roi_width, lane_width, num_lanes, fps
                        )
                        calculated_ppm = roi_width / (lane_width * num_lanes)
                        st.success(f"✅ Kalibrasyon oluşturuldu: {calculated_ppm:.2f} px/m")
            
            # Show current calibration
            if st.session_state.calibration:
                st.divider()
                st.subheader("Mevcut Kalibrasyon")
                st.write(f"**Piksel/Metre:** {st.session_state.calibration.pixels_per_meter:.2f}")
                st.write(f"**FPS:** {st.session_state.calibration.fps:.2f}")
    
    # Tab 4: Analysis
    with tab4:
        st.header("Analiz Çalıştırma")
        
        if st.session_state.video_path is None:
            st.warning("⚠️ Önce video yükleyin!")
        else:
            # Prepare ROI
            roi = None
            if st.session_state.multi_roi:
                # Multi-ROI will be passed separately
                pass
            elif st.session_state.roi_points:
                try:
                    roi = ROI(st.session_state.roi_points)
                except:
                    pass
            
            if st.button("🚀 Analizi Başlat", type="primary", use_container_width=True):
                with st.spinner("Analiz başlatılıyor..."):
                    try:
                        # Initialize components
                        vehicle_detector = YOLOVehicleDetector(
                            model_path="yolov8n.pt",
                            confidence_threshold=confidence,
                            debug=False
                        )
                        video_processor = OpenCVVideoProcessor()
                        
                        traffic_analyzer = TrafficAnalyzerService(
                            vehicle_detector=vehicle_detector,
                            video_processor=video_processor,
                            roi=roi,
                            multi_roi=st.session_state.multi_roi,
                            enable_smoothing=enable_smoothing,
                            smoothing_window=5,
                            drop_threshold=0.5,
                            enable_tracking=enable_tracking,
                            speed_calibration=st.session_state.calibration,
                            speed_limit_kmh=speed_limit,
                            enable_day_night_detection=False,
                            enable_weather_detection=False
                        )
                        
                        visualizer = VideoVisualizer(
                            show_confidence=True,
                            show_count=True,
                            show_speed=enable_tracking,
                            show_direction=enable_tracking,
                            show_tracking=enable_tracking
                        )
                        
                        # Progress tracking (minimal, only for callback)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Video display (full width)
                        st.markdown("### 🎥 Canlı Önizleme")
                        video_container = st.container()
                        with video_container:
                            st.markdown('<div class="video-container">', unsafe_allow_html=True)
                            video_placeholder = st.empty()
                            video_caption = st.empty()
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Track start time
                        start_time = time.time()
                        
                        def progress_callback(current_frame: int, total_frames: int):
                            if total_frames > 0:
                                progress = current_frame / total_frames
                                progress_bar.progress(progress)
                                status_text.text(f"İşleniyor: {current_frame}/{total_frames} frame ({progress*100:.1f}%)")
                        
                        # Output video setup
                        output_video_path = None
                        video_writer = None
                        if st.session_state.video_path:
                            cap_temp = cv2.VideoCapture(st.session_state.video_path)
                            fps = cap_temp.get(cv2.CAP_PROP_FPS)
                            width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            cap_temp.release()
                            
                            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                        
                        # Frame update frequency (show every N frames to avoid performance issues)
                        frame_update_frequency = max(1, skip_frames + 1) if skip_frames > 0 else 5
                        
                        def frame_callback(frame, vehicles, frame_stats, frame_number, roi=None, tracked_vehicles=None):
                            visualized_frame = visualizer.draw_vehicles(
                                frame, vehicles, frame_stats, roi=roi, tracked_vehicles=tracked_vehicles
                            )
                            
                            # Write to output video
                            if video_writer:
                                video_writer.write(visualized_frame)
                            
                            # Show video preview (update every N frames for performance)
                            if frame_number % frame_update_frequency == 0:
                                # Convert BGR to RGB for display
                                frame_rgb = cv2.cvtColor(visualized_frame, cv2.COLOR_BGR2RGB)
                                video_placeholder.image(frame_rgb, use_container_width=True)
                                
                                # Video caption (minimal)
                                video_caption.markdown(
                                    f"<div style='text-align: center; padding: 0.5rem; color: #666;'>Frame {frame_number} - Canlı Önizleme</div>",
                                    unsafe_allow_html=True
                                )
                            
                            return False
                        
                        # Run analysis
                        statistics = traffic_analyzer.analyze_video(
                            video_path=st.session_state.video_path,
                            progress_callback=progress_callback,
                            frame_callback=frame_callback,
                            skip_frames=skip_frames
                        )
                        
                        if video_writer:
                            video_writer.release()
                        
                        # Store results
                        st.session_state.analysis_results = statistics
                        st.session_state.output_video_path = output_video_path
                        
                        # Final update
                        progress_bar.progress(1.0)
                        status_text.text("✅ Analiz tamamlandı!")
                        st.success(" Analiz başarıyla tamamlandı!")
                        
                    except Exception as e:
                        st.error(f"❌ Hata: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # Tab 5: Results
    with tab5:
        st.header("Analiz Sonuçları")
        
        if st.session_state.analysis_results is None:
            st.info("👈 Önce analiz çalıştırın!")
        else:
            stats = st.session_state.analysis_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Toplam Araç", stats.total_vehicles)
            with col2:
                st.metric("Max Frame'de", stats.max_vehicles_in_frame)
            with col3:
                st.metric("Ortalama/Frame", f"{stats.average_vehicles_per_frame:.1f}")
            with col4:
                st.metric("Peak Yoğunluk", f"{stats.peak_density:.2f}%")
            
            # Detailed statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Araç Tipine Göre")
                if stats.vehicles_by_type:
                    for vtype, count in stats.vehicles_by_type.items():
                        if count > 0:
                            st.write(f"**{vtype.value.upper()}:** {count}")
                
                st.subheader("Araç Boyutuna Göre")
                if stats.vehicles_by_size:
                    for size, count in stats.vehicles_by_size.items():
                        if count > 0:
                            st.write(f"**{size.value.upper()}:** {count}")
            
            with col2:
                st.subheader("Yöne Göre")
                if stats.vehicles_by_direction:
                    for direction, count in stats.vehicles_by_direction.items():
                        if count > 0:
                            st.write(f"**{direction.value.upper()}:** {count}")
                
                st.subheader("Şerit Bazlı")
                if stats.vehicles_by_lane:
                    for lane_id, count in stats.vehicles_by_lane.items():
                        st.write(f"**{lane_id}:** {count}")
            
            # Speed statistics
            if enable_tracking and stats.total_tracked_vehicles > 0:
                st.divider()
                st.subheader("Hız İstatistikleri")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Takip Edilen Araç", stats.total_tracked_vehicles)
                with col2:
                    if stats.average_speed_kmh > 0:
                        st.metric("Ortalama Hız", f"{stats.average_speed_kmh:.1f} km/h")
                with col3:
                    if stats.max_speed_kmh > 0:
                        st.metric("Max Hız", f"{stats.max_speed_kmh:.1f} km/h")
                
                if stats.total_speed_violations > 0:
                    st.warning(f"⚠️ Hız İhlali: {stats.total_speed_violations} araç")
            
            # Download section
            st.divider()
            st.subheader("📥 İndirme")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON download
                json_str = json.dumps({
                    "total_vehicles": stats.total_vehicles,
                    "max_vehicles_in_frame": stats.max_vehicles_in_frame,
                    "average_vehicles_per_frame": stats.average_vehicles_per_frame,
                    "peak_density": stats.peak_density,
                    "vehicles_by_type": {vt.value: count for vt, count in stats.vehicles_by_type.items()},
                    "vehicles_by_size": {vs.value: count for vs, count in stats.vehicles_by_size.items()},
                    "vehicles_by_direction": {d.value: count for d, count in stats.vehicles_by_direction.items()},
                    "vehicles_by_lane": stats.vehicles_by_lane,
                    "total_tracked_vehicles": stats.total_tracked_vehicles,
                    "average_speed_kmh": stats.average_speed_kmh,
                    "max_speed_kmh": stats.max_speed_kmh,
                    "total_speed_violations": stats.total_speed_violations,
                }, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📄 JSON İndir",
                    data=json_str,
                    file_name="traffic_analysis_results.json",
                    mime="application/json"
                )
            
            with col2:
                # Video download
                if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
                    with open(st.session_state.output_video_path, 'rb') as f:
                        video_bytes = f.read()
                    st.download_button(
                        label="🎬 Video İndir",
                        data=video_bytes,
                        file_name="traffic_analysis_output.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.info("Video oluşturulmadı")


if __name__ == "__main__":
    main()


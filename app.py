import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Edge Lab Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Advanced edge detection & image filtering playground"
    }
)


# Custom CSS with improved styling
st.markdown("""
<style>
.stApp {
    background-color: #0f1117;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
}
div[data-testid="metric-container"] {
    background-color: #1e2130;
    border: 1px solid #2e3142;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- CACHED FUNCTIONS ----------------
@st.cache_data
def load_image(uploaded_file):
    """Load and cache uploaded images for better performance"""
    image = Image.open(uploaded_file)
    return np.array(image)


@st.cache_data
def preprocess_image(img, blur_val, enhance_contrast=False):
    """Preprocess with caching for performance"""
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Optional contrast enhancement
    if enhance_contrast:
        gray = cv2.equalizeHist(gray)
    
    # Apply blur
    if blur_val > 0:
        gray = cv2.GaussianBlur(gray, (blur_val|1, blur_val|1), 0)
    
    return gray


@st.cache_data
def apply_edge_detection(gray_img, method, th1, th2, ksize=3):
    """Apply edge detection with caching"""
    if method == "Sobel":
        gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)
        result = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
        
    elif method == "Canny":
        result = cv2.Canny(gray_img, th1, th2)
        
    elif method == "Laplacian":
        result = cv2.convertScaleAbs(cv2.Laplacian(gray_img, cv2.CV_64F, ksize=ksize))
        
    elif method == "Prewitt":
        kx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)
        ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
        gx = cv2.filter2D(gray_img, cv2.CV_32F, kx)
        gy = cv2.filter2D(gray_img, cv2.CV_32F, ky)
        result = cv2.convertScaleAbs(np.hypot(gx, gy))
        
    elif method == "Scharr":
        gx = cv2.Scharr(gray_img, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(gray_img, cv2.CV_64F, 0, 1)
        result = cv2.convertScaleAbs(np.hypot(gx, gy))
        
    elif method == "Laplacian of Gaussian":
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        result = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))
        
    return result


def convert_to_downloadable(img):
    """Convert numpy array to downloadable bytes"""
    is_success, buffer = cv2.imencode(".png", img)
    return io.BytesIO(buffer)


# ---------------- HEADER ----------------
st.title("🧪 Edge Lab Pro")
st.caption("Advanced edge detection & image filtering playground with real-time processing")


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

uploaded = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Supported formats: JPG, PNG, BMP, TIFF"
)

st.sidebar.divider()

# Edge Detection Method
edge_type = st.sidebar.selectbox(
    "🔍 Edge Detector",
    [
        "Canny",
        "Sobel",
        "Scharr",
        "Laplacian",
        "Prewitt",
        "Laplacian of Gaussian"
    ],
    help="Choose your edge detection algorithm"
)

st.sidebar.divider()

# Preprocessing
st.sidebar.subheader("🎛️ Preprocessing")
enhance_contrast = st.sidebar.checkbox("Enhance Contrast", value=False, 
                                      help="Apply histogram equalization")
blur = st.sidebar.slider("Gaussian Blur", 0, 15, 3, step=2,
                         help="Reduce noise before edge detection")

st.sidebar.divider()

# Edge Detection Parameters
st.sidebar.subheader("🎚️ Edge Parameters")

if edge_type == "Canny":
    th1 = st.sidebar.slider("Lower Threshold", 0, 255, 50,
                           help="Pixels below this are not edges")
    th2 = st.sidebar.slider("Upper Threshold", 0, 255, 150,
                           help="Pixels above this are strong edges")
else:
    th1 = st.sidebar.slider("Threshold 1", 0, 255, 50)
    th2 = st.sidebar.slider("Threshold 2", 0, 255, 150)

if edge_type in ["Sobel", "Laplacian"]:
    ksize = st.sidebar.select_slider("Kernel Size", options=[3, 5, 7], value=3,
                                     help="Larger kernels detect coarser edges")
else:
    ksize = 3

st.sidebar.divider()

# View Options
st.sidebar.subheader("👁️ View Options")
show_metrics = st.sidebar.checkbox("Show Metrics", value=True)
comparison_mode = st.sidebar.selectbox(
    "Display Mode",
    ["Side by Side", "Stacked", "Overlay"]
)


# ---------------- MAIN CONTENT ----------------
if uploaded:
    try:
        # Load image
        image = load_image(uploaded)
        
        # Process image
        gray = preprocess_image(image, blur, enhance_contrast)
        result = apply_edge_detection(gray, edge_type, th1, th2, ksize)
        
        # Metrics
        if show_metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Image Size", f"{image.shape[1]}×{image.shape[0]}")
            with col2:
                st.metric("Edge Pixels", f"{np.count_nonzero(result):,}")
            with col3:
                edge_density = (np.count_nonzero(result) / result.size) * 100
                st.metric("Edge Density", f"{edge_density:.2f}%")
            with col4:
                st.metric("Mean Intensity", f"{np.mean(result):.1f}")
            
            st.divider()
        
        # Display images based on mode
        if comparison_mode == "Side by Side":
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Original")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader(f"✨ {edge_type} Result")
                st.image(result, clamp=True, use_container_width=True)
                
        elif comparison_mode == "Stacked":
            st.subheader("📸 Original")
            st.image(image, use_container_width=True)
            st.subheader(f"✨ {edge_type} Result")
            st.image(result, clamp=True, use_container_width=True)
            
        elif comparison_mode == "Overlay":
            # Create colored overlay
            if len(result.shape) == 2:
                result_colored = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            else:
                result_colored = result
            
            if len(image.shape) == 2:
                image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_colored = image
                
            overlay = cv2.addWeighted(image_colored, 0.7, result_colored, 0.3, 0)
            
            st.subheader("🎨 Overlay View")
            st.image(overlay, use_container_width=True)
        
        # Download buttons
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.download_button(
                label="⬇️ Download Edge Map",
                data=convert_to_downloadable(result),
                file_name=f"edge_{edge_type.lower()}_{uploaded.name}",
                mime="image/png"
            )
        
        with col2:
            if comparison_mode == "Overlay":
                st.download_button(
                    label="⬇️ Download Overlay",
                    data=convert_to_downloadable(overlay),
                    file_name=f"overlay_{edge_type.lower()}_{uploaded.name}",
                    mime="image/png"
                )
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please try uploading a different image or adjusting the parameters.")

else:
    # Welcome screen
    st.info("👈 Upload an image from the sidebar to start exploring edge detection algorithms")
    
    with st.expander("ℹ️ About Edge Detection Methods"):
        st.markdown("""
        **Canny**: Multi-stage algorithm with noise reduction and edge tracking [web:9]
        
        **Sobel**: Gradient-based detection using convolution kernels
        
        **Scharr**: Similar to Sobel but with better rotational symmetry
        
        **Laplacian**: Second derivative method sensitive to noise
        
        **Prewitt**: Simple gradient-based method using 3×3 kernels [web:6]
        
        **Laplacian of Gaussian**: Combines Gaussian smoothing with Laplacian operator [web:6]
        """)
    
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        - Start with **Gaussian Blur** (3-5) to reduce noise [web:6]
        - **Canny** works best for most general purposes [web:9]
        - Adjust thresholds based on image contrast
        - Enable **Enhance Contrast** for low-contrast images [web:6]
        - Use larger kernel sizes for detecting coarse edges
        """)

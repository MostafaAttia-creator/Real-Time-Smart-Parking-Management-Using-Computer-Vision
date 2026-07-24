import streamlit as st
import cv2
import tempfile
from detector import process_video, process_image

st.set_page_config(
    page_title="Smart Parking Management System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Header
# ===========================

st.title("🚗 Smart Parking Management System")
st.markdown("### Real-Time Parking Slot Detection & Tracking")
st.divider()

# ===========================
# Sidebar
# ===========================

st.sidebar.title("⚙️ System Information")

st.sidebar.success("Model : YOLO11")

st.sidebar.info("Tracker : ByteTrack")

st.sidebar.markdown("### Classes")

st.sidebar.write("🟥 Busy")

st.sidebar.write("🟩 Empty")

st.sidebar.divider()

st.sidebar.markdown("### Upload")

uploaded_file = st.sidebar.file_uploader(
    "Choose Image or Video",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)

st.sidebar.divider()

start = st.sidebar.button(
    "▶ Start Detection",
    use_container_width=True
)

# ===========================
# Statistics
# ===========================

c1, c2, c3, c4 = st.columns(4)

occupied_metric = c1.empty()

free_metric = c2.empty()

total_metric = c3.empty()

occupancy_metric = c4.empty()

st.divider()

video_placeholder = st.empty()

# ===========================
# No File
# ===========================

if uploaded_file is None:

    st.info("Upload an image or video from the left panel.")

# ===========================
# Start
# ===========================

elif start:

    suffix = "." + uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        tmp.write(uploaded_file.read())

        temp_path = tmp.name

    # =======================
    # IMAGE
    # =======================

    if uploaded_file.type.startswith("image"):

        frame, occupied, free, total, occ = process_image(
            temp_path
        )

        occupied_metric.metric(
            "🟥 Occupied",
            occupied
        )

        free_metric.metric(
            "🟩 Free",
            free
        )

        total_metric.metric(
            "📌 Total",
            total
        )

        occupancy_metric.metric(
            "📊 Occupancy",
            f"{occ:.1f}%"
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        video_placeholder.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

    # =======================
    # VIDEO
    # =======================

    else:

        for frame, occupied, free, total, occ in process_video(
            temp_path
        ):

            occupied_metric.metric(
                "🟥 Occupied",
                occupied
            )

            free_metric.metric(
                "🟩 Free",
                free
            )

            total_metric.metric(
                "📌 Total",
                total
            )

            occupancy_metric.metric(
                "📊 Occupancy",
                f"{occ:.1f}%"
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            video_placeholder.image(
                frame,
                channels="RGB",
                use_container_width=True
            )

st.divider()

st.caption("Smart Parking Management System © 2026")
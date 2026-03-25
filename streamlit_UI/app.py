import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Table, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os

# -------------------------------
# LOAD MODELS
# -------------------------------
bogie_model = YOLO("/Users/arjavpatil/Documents/Review2/ffinal_system/best (5).pt")
panel_model = YOLO("/Users/arjavpatil/Documents/Review2/ffinal_system/best.pt")
spring_model = YOLO("/Users/arjavpatil/Documents/Review2/ffinal_system/best (1).pt")

st.title("🚆 Bogie Inspection System")

uploaded_file = st.file_uploader("Upload Bogie Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # -------------------------------
    # READ IMAGE
    # -------------------------------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale (as required)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    original = img.copy()

    # -------------------------------
    # DETECT BOGIE
    # -------------------------------
    bogie_results = bogie_model(img, conf=0.3)

    bogie_data = []

    for i, box in enumerate(bogie_results[0].boxes):

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 🔴 Draw bogie box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Crop bogie
        crop = original[y1:y2, x1:x2]

        # -------------------------------
        # PANEL DETECTION (RED)
        # -------------------------------
        panel_results = panel_model(crop, conf=0.3)

        panel_status = "None"

        for pbox in panel_results[0].boxes:

            px1, py1, px2, py2 = map(int, pbox.xyxy[0])

            # 🔥 Map back to original image
            px1 += x1
            px2 += x1
            py1 += y1
            py2 += y1

            # 🔴 Draw panel box
            cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)

            cls_id = int(pbox.cls[0])
            panel_status = panel_model.names[cls_id]

            cv2.putText(img, panel_status, (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # -------------------------------
        # SPRING DETECTION (GREEN)
        # -------------------------------
        spring_results = spring_model(crop, conf=0.3)

        spring_count = 0

        for sbox in spring_results[0].boxes:

            sx1, sy1, sx2, sy2 = map(int, sbox.xyxy[0])

            # 🔥 Map back
            sx1 += x1
            sx2 += x1
            sy1 += y1
            sy2 += y1

            # 🟢 Draw spring box
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

            cv2.putText(img, "Spring", (sx1, sy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            spring_count += 1

        bogie_data.append({
            "id": f"Bogie {i+1}",
            "panel": panel_status,
            "springs": spring_count,
            "coords": (x1, y1, x2, y2)
        })

    # -------------------------------
    # SHOW OUTPUT IMAGE
    # -------------------------------
    st.image(img, caption="Detection Output", use_container_width=True)

    # -------------------------------
    # GENERATE PDF
    # -------------------------------
    if st.button("Generate Report"):

        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        doc = SimpleDocTemplate(pdf_path)
        elements = []
        styles = getSampleStyleSheet()

        for bogie in bogie_data:

            elements.append(Paragraph(f"<b>{bogie['id']}</b>", styles["Title"]))
            elements.append(Spacer(1, 10))

            # Crop annotated bogie region from FULL annotated image
            x1, y1, x2, y2 = bogie["coords"]
            bogie_img = img[y1:y2, x1:x2]

            temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            cv2.imwrite(temp_img_path, bogie_img)

            elements.append(RLImage(temp_img_path, width=450, height=250))
            elements.append(Spacer(1, 10))

            table_data = [
                ["Bogie Number", "Panel Status", "Spring Count"],
                [bogie["id"], bogie["panel"], str(bogie["springs"])]
            ]

            table = Table(table_data)
            table.setStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.grey),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("GRID", (0,0), (-1,-1), 1, colors.black)
            ])

            elements.append(table)
            elements.append(PageBreak())

        doc.build(elements)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download Report",
                f,
                file_name="bogie_report.pdf",
                mime="application/pdf"
            )

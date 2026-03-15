import streamlit as st
import numpy as np
import pickle
import io
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load trained model
model = pickle.load(open("model/diabetes_model.pkl", "rb"))

st.title("AI Diabetes Prediction System")

st.write("Enter patient medical details")

preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

prediction_result = None

if st.button("Predict Diabetes"):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        prediction_result = "High Risk of Diabetes"
        st.error(prediction_result)
    else:
        prediction_result = "Low Risk of Diabetes"
        st.success(prediction_result)


def create_pdf():

    buffer = io.BytesIO()

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("AI Diabetes Prediction Report", styles['Title']))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Pregnancies: {preg}", styles['Normal']))
    elements.append(Paragraph(f"Glucose Level: {glucose}", styles['Normal']))
    elements.append(Paragraph(f"Blood Pressure: {bp}", styles['Normal']))
    elements.append(Paragraph(f"Skin Thickness: {skin}", styles['Normal']))
    elements.append(Paragraph(f"Insulin Level: {insulin}", styles['Normal']))
    elements.append(Paragraph(f"BMI: {bmi}", styles['Normal']))
    elements.append(Paragraph(f"Diabetes Pedigree Function: {dpf}", styles['Normal']))
    elements.append(Paragraph(f"Age: {age}", styles['Normal']))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Prediction Result: {prediction_result}", styles['Heading2']))

    pdf = SimpleDocTemplate(buffer)
    pdf.build(elements)

    buffer.seek(0)
    return buffer


if prediction_result is not None:

    pdf_file = create_pdf()

    st.download_button(
        label="Download Patient Report PDF",
        data=pdf_file,
        file_name="diabetes_patient_report.pdf",
        mime="application/pdf"
    )
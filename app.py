import streamlit as st
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

# --- CSS Styling for Professional Look ---
st.markdown("""
    <style>
    body {
        background-color: #0B2545;
        color: #FFFFFF;
    }
    .card {
        background-color: #E8F1F5;
        color: #0B2545;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #E8F1F5;  /* Changed to light background */
        color: #0B2545;  /* Changed to dark text */
        border-radius: 8px;
        padding: 0.5em 1em;
        margin-top: 5px;
        font-weight: 500;
        border: 1px solid #2A5C9E;  /* Added border for definition */
    }
    .stButton>button:hover {
        background-color: #D4E9F7;  /* Slightly darker on hover */
        color: #0B2545;
    }
    .stRadio > div {
        flex-direction: column;
    }
    details {
        background-color: #E8F1F5;  /* Changed to light background */
        color: #0B2545;  /* Changed to dark text */
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    summary {
        font-weight: bold;
        cursor: pointer;
        color: #0B2545;  /* Ensure summary text is dark */
    }
    img {
        max-width: 100%;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .result-box {
        background-color: #D4E9F7;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #0B2545;
        border-left: 4px solid #2A5C9E;
    }
    .stTextInput>div>div>input {
        background-color: white;
        color: #0B2545;  /* Dark text in input */
    }
    .stSelectbox>div>div>select {
        background-color: white;
        color: #0B2545;  /* Dark text in select */
    }
    /* Fix expander styling */
    .streamlit-expanderHeader {
        background-color: #E8F1F5 !important;
        color: #0B2545 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_age" not in st.session_state:
    st.session_state.patient_age = ""
if "patient_gender" not in st.session_state:
    st.session_state.patient_gender = ""

# --- Tabs ---
tabs = ["AI Diagnosis", "About DR", "Symptoms Guide", "Quiz", "Generate Report"]
tab = st.sidebar.radio("Navigation", tabs)

# -------------------- AI Diagnosis Tab --------------------
if tab == "AI Diagnosis":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🤖 AI Diagnosis")
    
    # Image Upload Section
    st.subheader("Upload Retinal Image")
    uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        st.image(image, caption="Uploaded Retinal Image", use_column_width=True)
        
        # Analyze Button
        if st.button("🔍 Analyze Image", key="analyze_btn"):
            with st.spinner("Analyzing image..."):
                # Placeholder for model prediction
                # In actual implementation, you would call your model here
                import time
                time.sleep(2)  # Simulate processing
                
                # Mock results (replace with actual model output)
                st.session_state.diagnosis_result = {
                    "stage": "Moderate Non-Proliferative DR",
                    "confidence": 0.87,
                    "description": "The retina shows signs of moderate diabetic retinopathy with microaneurysms and dot/blot hemorrhages."
                }
                st.session_state.processed_image = image  # Replace with actual GradCAM image
            
            st.success("✅ Analysis Complete!")
        
        # Show results if available
        if st.session_state.diagnosis_result:
            st.markdown("---")
            
            # Create two columns for the options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 See Results", key="see_results_btn", use_container_width=True):
                    st.session_state.show_results = True
                    st.session_state.show_processed = False
            
            with col2:
                if st.button("🖼️ See Processed Image", key="see_processed_btn", use_container_width=True):
                    st.session_state.show_processed = True
                    st.session_state.show_results = False
            
            # Display Results Tab
            if st.session_state.get("show_results", False):
                st.markdown("### Diagnosis Results")
                result = st.session_state.diagnosis_result
                
                st.markdown(f"""
                <div class="result-box">
                    <h3>🔬 Detected Stage: {result['stage']}</h3>
                    <p><strong>Confidence:</strong> {result['confidence']*100:.1f}%</p>
                    <p><strong>Description:</strong> {result['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Standard diagnosis based on stage
                st.markdown("### 📋 Standard Diagnosis & Recommendations")
                
                stage_recommendations = {
                    "Mild Non-Proliferative DR": {
                        "findings": "Microaneurysms present in the retina",
                        "recommendations": [
                            "Monitor blood sugar levels closely",
                            "Schedule eye exams every 6-12 months",
                            "Maintain HbA1c below 7%",
                            "Control blood pressure"
                        ]
                    },
                    "Moderate Non-Proliferative DR": {
                        "findings": "Multiple microaneurysms, dot/blot hemorrhages, and some vessel blockage detected",
                        "recommendations": [
                            "Eye exams every 3-6 months",
                            "Strict glycemic control required",
                            "Consider laser treatment consultation",
                            "Monitor for macular edema",
                            "Control blood pressure and cholesterol"
                        ]
                    },
                    "Severe Non-Proliferative DR": {
                        "findings": "Extensive hemorrhages, cotton wool spots, and venous beading observed",
                        "recommendations": [
                            "Immediate ophthalmologist consultation",
                            "Eye exams every 2-3 months",
                            "Laser photocoagulation treatment recommended",
                            "Intensive blood sugar management",
                            "Monitor for progression to PDR"
                        ]
                    },
                    "Proliferative DR": {
                        "findings": "Abnormal new blood vessel growth (neovascularization) detected",
                        "recommendations": [
                            "URGENT: Immediate treatment required",
                            "Panretinal photocoagulation (PRP) laser surgery",
                            "Anti-VEGF injections may be needed",
                            "Monthly follow-up appointments",
                            "High risk of vision loss - immediate action needed"
                        ]
                    }
                }
                
                stage_info = stage_recommendations.get(result['stage'], {
                    "findings": "Findings require professional evaluation",
                    "recommendations": ["Consult with an ophthalmologist"]
                })
                
                st.markdown(f"**Clinical Findings:** {stage_info['findings']}")
                st.markdown("**Recommendations:**")
                for rec in stage_info['recommendations']:
                    st.markdown(f"- {rec}")
            
            # Display Processed Image Tab
            if st.session_state.get("show_processed", False):
                st.markdown("### Processed Image with GradCAM")
                st.info("GradCAM visualization shows the areas of the retina that contributed most to the diagnosis.")
                if st.session_state.processed_image:
                    st.image(st.session_state.processed_image, caption="GradCAM Heatmap", use_column_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- About DR Tab --------------------
elif tab == "About DR":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("ℹ️ About Diabetic Retinopathy")

    stages = [
        ("Mild Non-Proliferative DR", "Presence of microaneurysms, small red dots in the retina.", "stage1.jpg"),
        ("Moderate Non-Proliferative DR", "More extensive microaneurysms, dot/blot hemorrhages, some vessel blockage.", "stage2.jpg"),
        ("Severe Non-Proliferative DR", "Many hemorrhages, cotton wool spots, venous beading; high risk of progression.", "stage3.jpg"),
        ("Proliferative DR", "Abnormal new vessels (neovascularization), high risk of bleeding and vision loss.", "stage4.jpg")
    ]

    for stage, desc, img_path in stages:
        with st.expander(stage):
            st.write(desc)
            try:
                st.image(img_path, use_column_width=True)
            except:
                st.warning(f"Image {img_path} not found. Please make sure it's in the same folder as app.py.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Symptoms Guide Tab --------------------
elif tab == "Symptoms Guide":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("👁️ Symptoms Guide")
    st.write("Learn about diabetic retinopathy symptoms and tips for prevention:")

    symptoms = [
        ("Floaters or dark spots in vision", "Small shadows or dark strings appear in your vision due to bleeding in the retina."),
        ("Blurred or distorted vision", "Blood vessel leakage causes distorted vision."),
        ("Poor night vision", "Difficulty seeing at night or in dim light conditions."),
        ("Eye pain or pressure", "Usually in advanced stages, can signal complications."),
        ("Gradual vision loss", "Slow loss of vision over time, often unnoticed initially.")
    ]

    for symptom, explanation in symptoms:
        st.markdown(f"<details><summary>{symptom}</summary>{explanation}</details>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Quiz Tab --------------------
elif tab == "Quiz":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("❓ Diabetic Retinopathy Quiz")

    questions = [
        {"q": "Which part of the eye does diabetic retinopathy primarily affect?", "opts": ["Cornea", "Lens", "Retina", "Optic nerve"], "ans": 2, "exp": "Correct! DR affects retina vessels."},
        {"q": "Which is a common early symptom of diabetic retinopathy?", "opts": ["Sudden blindness", "Floaters or spots", "Eye pain", "Double vision"], "ans": 1, "exp": "Floaters are early signs due to micro-bleeds."},
        {"q": "Which of these increases the likelihood of DR?", "opts": ["Short duration of diabetes", "Poor blood sugar control", "Low blood pressure", "Low cholesterol"], "ans": 1, "exp": "High blood sugar damages retinal vessels."},
        {"q": "How often should a diabetic person get an eye exam?", "opts": ["Every 5 years", "Only if vision declines", "Annually", "Never if no symptoms"], "ans": 2, "exp": "Annual exams catch DR early."},
        {"q": "Fluid in the center of retina is called?", "opts": ["Retinal detachment", "Macular edema", "Glaucoma", "Cataract"], "ans": 1, "exp": "Macular edema results from vessel leakage."},
        {"q": "Which of these is a lifestyle change to reduce risk?", "opts": ["Smoking more", "Exercising regularly", "Avoiding eye exams", "Increasing sugar intake"], "ans": 1, "exp": "Exercise helps control blood sugar."},
        {"q": "Severe DR treatment?", "opts": ["Anti-VEGF injections", "Eyeglasses prescription", "Antibiotic drops", "Vitamin C supplements"], "ans": 0, "exp": "Anti-VEGF reduces abnormal vessel growth."},
        {"q": "Early DR always has symptoms?", "opts": ["True", "False", "Only type 1", "Only type 2"], "ans": 1, "exp": "False. Early DR often asymptomatic."},
        {"q": "Underlying cause of DR?", "opts": ["High blood sugar", "Low blood sugar", "Kidney failure", "High calcium"], "ans": 0, "exp": "High sugar damages retinal vessels."},
        {"q": "Which of these is not recommended to manage DR risk?", "opts": ["Regular eye exams", "Manage blood pressure", "Quit smoking", "Skip sugar monitoring"], "ans": 3, "exp": "Monitoring sugar is essential."},
    ]

    for i, q in enumerate(questions):
        st.markdown(f"**{i+1}. {q['q']}**")
        choice = st.radio("", q["opts"], key=f"q{i}")
        if st.button(f"Submit Q{i+1}", key=f"btn{i}"):
            if st.session_state.get(f"answered_{i}", False):
                st.warning("You already answered this question!")
            else:
                if q["opts"].index(choice) == q["ans"]:
                    st.success("✅ Correct!")
                    st.info(q["exp"])
                    st.session_state.quiz_score += 1
                else:
                    st.error(f"❌ Incorrect. Correct: **{q['opts'][q['ans']]}**")
                    st.info(q["exp"])
                st.session_state[f"answered_{i}"] = True

    st.write("---")
    st.write(f"**Your total score: {st.session_state.quiz_score} / {len(questions)}**")
    if st.button("Reset Quiz"):
        st.session_state.quiz_score = 0
        for i in range(len(questions)):
            st.session_state[f"answered_{i}"] = False

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Generate Report Tab --------------------
elif tab == "Generate Report":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📝 Generate Medical Report")
    
    # Patient Information Form
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Full Name", value=st.session_state.patient_name, key="name_input")
        patient_age = st.text_input("Age", value=st.session_state.patient_age, key="age_input")
    
    with col2:
        patient_gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"], 
                                      index=0 if not st.session_state.patient_gender else 
                                      ["Select", "Male", "Female", "Other"].index(st.session_state.patient_gender))
    
    # Update session state
    st.session_state.patient_name = patient_name
    st.session_state.patient_age = patient_age
    if patient_gender != "Select":
        st.session_state.patient_gender = patient_gender
    
    st.markdown("---")
    
    # Generate Report Button
    if st.button("📄 Generate & Download Report", type="primary", use_container_width=True):
        if not patient_name or not patient_age or patient_gender == "Select":
            st.error("⚠️ Please fill in all patient information fields.")
        elif not st.session_state.diagnosis_result:
            st.error("⚠️ Please complete an AI diagnosis first before generating a report.")
        else:
            # Generate HTML report
            result = st.session_state.diagnosis_result
            
            # Get stage-specific recommendations
            stage_recommendations = {
                "Mild Non-Proliferative DR": {
                    "findings": "Microaneurysms present in the retina",
                    "treatment": "Monitor blood sugar levels closely, schedule eye exams every 6-12 months, maintain HbA1c below 7%, and control blood pressure."
                },
                "Moderate Non-Proliferative DR": {
                    "findings": "Multiple microaneurysms, dot/blot hemorrhages, and some vessel blockage detected",
                    "treatment": "Eye exams every 3-6 months, strict glycemic control required, consider laser treatment consultation, monitor for macular edema, and control blood pressure and cholesterol."
                },
                "Severe Non-Proliferative DR": {
                    "findings": "Extensive hemorrhages, cotton wool spots, and venous beading observed",
                    "treatment": "Immediate ophthalmologist consultation required. Eye exams every 2-3 months, laser photocoagulation treatment recommended, intensive blood sugar management, and close monitoring for progression to PDR."
                },
                "Proliferative DR": {
                    "findings": "Abnormal new blood vessel growth (neovascularization) detected",
                    "treatment": "URGENT: Immediate treatment required. Panretinal photocoagulation (PRP) laser surgery, Anti-VEGF injections may be needed, monthly follow-up appointments. High risk of vision loss - immediate action needed."
                }
            }
            
            stage_info = stage_recommendations.get(result['stage'], {
                "findings": "Findings require professional evaluation",
                "treatment": "Please consult with an ophthalmologist for detailed treatment plan."
            })
            
            # Convert image to base64
            img_base64 = ""
            if st.session_state.uploaded_image:
                buffered = BytesIO()
                st.session_state.uploaded_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        padding: 40px;
                        color: #333;
                    }}
                    .header {{
                        text-align: center;
                        border-bottom: 3px solid #2A5C9E;
                        padding-bottom: 20px;
                        margin-bottom: 30px;
                    }}
                    .header h1 {{
                        color: #2A5C9E;
                        margin: 0;
                    }}
                    .section {{
                        margin: 20px 0;
                        padding: 15px;
                        background-color: #f5f5f5;
                        border-radius: 8px;
                    }}
                    .section h2 {{
                        color: #2A5C9E;
                        border-bottom: 2px solid #2A5C9E;
                        padding-bottom: 10px;
                    }}
                    .info-row {{
                        margin: 10px 0;
                    }}
                    .label {{
                        font-weight: bold;
                        color: #555;
                    }}
                    .diagnosis-box {{
                        background-color: #E8F1F5;
                        padding: 15px;
                        border-left: 4px solid #2A5C9E;
                        margin: 15px 0;
                    }}
                    .image-container {{
                        text-align: center;
                        margin: 20px 0;
                    }}
                    .image-container img {{
                        max-width: 500px;
                        border: 2px solid #ddd;
                        border-radius: 8px;
                    }}
                    .footer {{
                        margin-top: 40px;
                        text-align: center;
                        font-size: 12px;
                        color: #777;
                        border-top: 1px solid #ddd;
                        padding-top: 20px;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🏥 DIABETIC RETINOPATHY DIAGNOSIS REPORT</h1>
                    <p>AI-Powered Retinal Analysis</p>
                </div>
                
                <div class="section">
                    <h2>👤 Patient Information</h2>
                    <div class="info-row"><span class="label">Name:</span> {patient_name}</div>
                    <div class="info-row"><span class="label">Age:</span> {patient_age} years</div>
                    <div class="info-row"><span class="label">Gender:</span> {patient_gender}</div>
                    <div class="info-row"><span class="label">Report Date:</span> {datetime.now().strftime("%B %d, %Y")}</div>
                    <div class="info-row"><span class="label">Report Time:</span> {datetime.now().strftime("%I:%M %p")}</div>
                </div>
                
                <div class="section">
                    <h2>🔬 AI Diagnosis Results</h2>
                    <div class="diagnosis-box">
                        <h3 style="margin-top: 0; color: #2A5C9E;">Detected Stage: {result['stage']}</h3>
                        <div class="info-row"><span class="label">Confidence Level:</span> {result['confidence']*100:.1f}%</div>
                        <div class="info-row"><span class="label">Clinical Findings:</span> {stage_info['findings']}</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>💊 Recommended Treatment</h2>
                    <p>{stage_info['treatment']}</p>
                </div>
                
                <div class="section">
                    <h2>📊 Quiz Performance</h2>
                    <div class="info-row"><span class="label">Score:</span> {st.session_state.quiz_score} / 10</div>
                    <div class="info-row"><span class="label">Percentage:</span> {(st.session_state.quiz_score/10)*100:.0f}%</div>
                </div>
                
                {"<div class='section'><h2>📷 Retinal Image</h2><div class='image-container'><img src='data:image/png;base64," + img_base64 + "' alt='Retinal Image'/></div></div>" if img_base64 else ""}
                
                <div class="footer">
                    <p><strong>Disclaimer:</strong> This report is generated by an AI system and should not replace professional medical advice. 
                    Please consult with a qualified ophthalmologist for proper diagnosis and treatment.</p>
                    <p>Report generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
                </div>
            </body>
            </html>
            """
            
            # Create download button
            st.success("✅ Report generated successfully!")
            st.download_button(
                label="⬇️ Download Report (HTML)",
                data=html_content,
                file_name=f"DR_Report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                use_container_width=True
            )
            
            st.info("💡 Tip: Open the downloaded HTML file in any web browser to view your complete report.")
    
    st.markdown('</div>', unsafe_allow_html=True)

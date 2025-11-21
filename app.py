import streamlit as st
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

# Import model utilities
from model_utils import (
    load_model, preprocess_image, make_gradcam_heatmap, 
    overlay_gradcam, predict_dr_stage, get_stage_recommendations,
    create_comparison_image
)

# Page config
st.set_page_config(
    page_title="DR Detection System",
    page_icon="",
    layout="wide"
)

# --- CSS Styling for Professional Look (Dark Mode Compatible) ---
st.markdown("""
    <style>
    /* Light Mode (Default) */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        background-attachment: fixed;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e9ecef 0%, #dee2e6 100%);
    }

    body {
        background-color: #0B2545;
        color: #0B2545;
    }

    .card {
        background-color: #E8F1F5;
        color: #0B2545;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }

    .stButton > button {
        background-color: #E8F1F5;
        color: #0B2545;
        border-radius: 8px;
        padding: 0.5em 1em;
        margin-top: 5px;
        font-weight: 500;
        border: 1px solid #2A5C9E;
    }

    .stButton > button:hover {
        background-color: #D4E9F7;
        color: #0B2545;
    }

    .stRadio > div {
        flex-direction: column;
    }

    details {
        background-color: #E8F1F5;
        color: #0B2545;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }

    summary {
        font-weight: bold;
        cursor: pointer;
        color: #0B2545;
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

    .result-box.critical {
        background-color: #FFE5E5;
        border-left: 4px solid #D32F2F;
        color: #333;
    }

    .result-box.high {
        background-color: #FFF3E0;
        border-left: 4px solid #F57C00;
        color: #333;
    }

    .result-box.medium {
        background-color: #FFF9C4;
        border-left: 4px solid #FBC02D;
        color: #333;
    }

    .result-box.low {
        background-color: #E8F5E9;
        border-left: 4px solid #388E3C;
        color: #333;
    }

    .stTextInput > div > div > input {
        background-color: white;
        color: #0B2545;
    }

    .stSelectbox > div > div > select {
        background-color: white;
        color: #0B2545;
    }

    .streamlit-expanderHeader {
        background-color: #E8F1F5 !important;
        color: #0B2545 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #E8F1F5;
        color: #0B2545;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2A5C9E;
        color: white;
    }

    h1, h2, h3, p {
        color: #0B2545;
    }

    /* Light mode text for info boxes */
    .stInfo, .stWarning, .stError, .stSuccess {
        border-radius: 8px;
    }

    /* DARK MODE STYLES */
    @media (prefers-color-scheme: dark) {
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
            background-attachment: fixed;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1f2e 0%, #252d3d 100%);
        }

        body {
            background-color: #f0f0f0;
            color: #e0e0e0;
        }

        .card {
            background-color: #1e3a5f;
            color: #e0e0e0;
            box-shadow: 0 4px 15px rgba(100, 181, 246, 0.2);
        }

        .stButton > button {
            background-color: #1e3a5f;
            color: #64B5F6;
            border: 1px solid #64B5F6;
        }

        .stButton > button:hover {
            background-color: #2a7dd9;
            color: white;
        }

        details {
            background-color: #1e3a5f;
            color: #e0e0e0;
        }

        summary {
            color: #64B5F6;
        }

        .result-box {
            background-color: #1e3a5f;
            color: #e0e0e0;
            border-left: 4px solid #64B5F6;
        }

        .result-box.critical {
            background-color: #4a2020;
            color: #ff9999;
            border-left: 4px solid #ff6b6b;
        }

        .result-box.high {
            background-color: #4a3820;
            color: #ffb74d;
            border-left: 4px solid #ffb74d;
        }

        .result-box.medium {
            background-color: #4a4620;
            color: #ffd54f;
            border-left: 4px solid #ffd54f;
        }

        .result-box.low {
            background-color: #1b3a1b;
            color: #81c784;
            border-left: 4px solid #81c784;
        }

        .stTextInput > div > div > input {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border-color: #64B5F6 !important;
        }

        .stSelectbox > div > div > select {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border-color: #64B5F6 !important;
        }

        .streamlit-expanderHeader {
            background-color: #1e3a5f !important;
            color: #e0e0e0 !important;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #1e3a5f;
            color: #e0e0e0;
        }

        .stTabs [aria-selected="true"] {
            background-color: #64B5F6;
            color: #0B2545;
        }

        h1, h2, h3 {
            color: #64B5F6;
        }

        p, span {
            color: #e0e0e0;
        }
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
if "gradcam_image" not in st.session_state:
    st.session_state.gradcam_image = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_age" not in st.session_state:
    st.session_state.patient_age = ""
if "patient_gender" not in st.session_state:
    st.session_state.patient_gender = ""
if "model" not in st.session_state:
    st.session_state.model = None

# Load model once
@st.cache_resource
def get_model():
    """Load model with caching"""
    return load_model('dr_model.h5')

# --- Tabs ---
tabs = ["AI Diagnosis", "About DR", "Symptoms Guide", "Quiz", "Generate Report"]
tab = st.sidebar.radio("Navigation", tabs)

# -------------------- AI Diagnosis Tab --------------------
if tab == "AI Diagnosis":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(" AI-Powered Diabetic Retinopathy Diagnosis")
    
    # Check if model is loaded
    if st.session_state.model is None:
        with st.spinner("Loading AI model... This may take a moment."):
            st.session_state.model = get_model()
        
        if st.session_state.model is None:
            st.error(" Failed to load the AI model. Please ensure 'dr_model.h5' is in the app directory.")
            st.stop()
        else:
            st.success(" AI model loaded successfully!")
    
    # Image Upload Section
    st.subheader(" Upload Retinal Image")
    st.info(" Upload a fundus photograph (retinal image) for AI analysis. Supported formats: JPG, JPEG, PNG")
    
    uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display uploaded image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert to RGB if necessary
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        image_np = image_np.astype(np.uint8)
        st.session_state.uploaded_image = image
        st.session_state.original_image = image_np.copy()
        
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption=" Uploaded Retinal Image", use_container_width=200)
        
        st.markdown("---")
        
        # Analyze Button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(" Analyze Image", type="primary", use_container_width=200):
                with st.spinner(" Processing image and running AI analysis..."):
                    try:
                        # Preprocess image
                        processed_img = preprocess_image(image_np, target_size=(320, 320))
                        st.session_state.processed_image = (processed_img * 255).astype(np.uint8)
                        
                        # Get prediction
                        stage_name, confidence, raw_pred = predict_dr_stage(processed_img, st.session_state.model)
                        
                        # Generate GradCAM
                        img_array = np.expand_dims(processed_img, axis=0)
                        
                        # Try to find the last conv layer
                        last_conv_layer = None
                        for layer in reversed(st.session_state.model.layers):
                            if 'conv' in layer.name.lower():
                                last_conv_layer = layer.name
                                break
                        
                        if last_conv_layer:
                            heatmap = make_gradcam_heatmap(img_array, st.session_state.model, last_conv_layer)
                            gradcam_overlay = overlay_gradcam(st.session_state.processed_image, heatmap)
                            st.session_state.gradcam_image = gradcam_overlay
                        else:
                            st.session_state.gradcam_image = st.session_state.processed_image
                        
                        # Get stage recommendations
                        stage_info = get_stage_recommendations(stage_name)
                        
                        # Store results
                        st.session_state.diagnosis_result = {
                            "stage": stage_name,
                            "confidence": confidence,
                            "raw_prediction": raw_pred,
                            "findings": stage_info["findings"],
                            "recommendations": stage_info["recommendations"],
                            "severity": stage_info["severity"]
                        }
                        
                        st.success(" Analysis Complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f" Error during analysis: {str(e)}")
                        st.info("Please try uploading a different image or check if the model file is correct.")
        
        # Show results if available
        if st.session_state.diagnosis_result:
            st.markdown("---")
            st.subheader(" Analysis Results")
            
            # Create tabs for results
            tab1, tab2 = st.tabs([" Diagnosis Results", " Processed Images"])
            
            with tab1:
                result = st.session_state.diagnosis_result
                severity_class = result.get("severity", "low")
                
                # Main diagnosis box
                st.markdown(f"""
                <div class="result-box {severity_class}">
                    <h2 style="margin-top: 0;"> Detected Stage: {result['stage']}</h2>
                    <p style="font-size: 18px;"><strong>Confidence Level:</strong> {result['confidence']*100:.1f}%</p>
                    <p style="font-size: 16px;"><strong>Raw Prediction Value:</strong> {result['raw_prediction']:.3f} (Scale: 0-4)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Clinical Findings
                st.markdown("###  Clinical Findings")
                st.info(result['findings'])
                
                # Recommendations
                st.markdown("###  Recommended Actions")
                
                if severity_class in ["critical", "high"]:
                    st.error(" **URGENT ACTION REQUIRED**")
                elif severity_class == "medium":
                    st.warning(" **Medical Attention Recommended**")
                else:
                    st.success(" **Continue Regular Monitoring**")
                
                for i, rec in enumerate(result['recommendations'], 1):
                    if rec.startswith("") or rec.startswith(""):
                        st.error(f"{i}. {rec}")
                    else:
                        st.write(f"{i}. {rec}")
                
                # Additional Information
                with st.expander(" Understanding Your Results"):
                    st.write("""
                    **About the AI Analysis:**
                    - Our AI model uses deep learning to analyze retinal images
                    - The model was trained on thousands of fundus photographs
                    - Confidence level indicates how certain the model is about its prediction
                    - Raw prediction value shows the continuous output before rounding to a stage
                    
                    **Important Notes:**
                    - This is a screening tool and NOT a replacement for professional diagnosis
                    - Always consult with a qualified ophthalmologist for proper evaluation
                    - Early detection and treatment can prevent vision loss
                    - Regular eye exams are crucial for diabetic patients
                    """)
            
            with tab2:
                st.markdown("###  Image Analysis Visualization")
                
                # Show three images side by side
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Original Image**")
                    st.image(st.session_state.original_image, use_container_width=True)
                    st.caption(" As uploaded")
                
                with col2:
                    st.markdown("**Preprocessed Image**")
                    st.image(st.session_state.processed_image, use_container_width=True)
                    st.caption("🔧 After circle crop & enhancement")
                
                with col3:
                    st.markdown("**GradCAM Heatmap**")
                    st.image(st.session_state.gradcam_image, use_container_width=True)
                    st.caption(" AI attention areas")
                
                st.markdown("---")
                
                with st.expander(" Understanding GradCAM Visualization"):
                    st.write("""
                    **What is GradCAM?**
                    
                    Gradient-weighted Class Activation Mapping (GradCAM) is a visualization technique that shows 
                    which regions of the retinal image were most important for the AI's decision.
                    
                    **How to interpret the heatmap:**
                    -  **Red/Yellow areas**: Regions that strongly influenced the AI's diagnosis
                    -  **Blue/Purple areas**: Regions with less influence on the prediction
                    - These highlighted areas often correspond to lesions, hemorrhages, or other pathological features
                    
                    **Preprocessing steps applied:**
                    1. Circle crop to focus on retinal area
                    2. Resize to standard dimensions (320x320)
                    3. Ben Graham's preprocessing for contrast enhancement
                    4. Normalization for neural network input
                    """)
    
    else:
        # Show instructions when no image is uploaded
        st.info("""
         **Please upload a retinal fundus image to begin analysis**
        
        **Tips for best results:**
        - Use clear, well-lit fundus photographs
        - Ensure the entire retina is visible
        - Avoid images with excessive blur or artifacts
        - JPG, JPEG, or PNG formats are supported
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- About DR Tab --------------------
elif tab == "About DR":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(" About Diabetic Retinopathy")
    
    st.write("""
    Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes. 
    It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).
    """)

    stages = [
        ("Mild Non-Proliferative DR", "Microaneurysms present - small bulges in retinal blood vessels.", "stage1.jpg"),
        ("Moderate Non-Proliferative DR", "More extensive microaneurysms, dot/blot hemorrhages, some vessel blockage.", "stage2.jpg"),
        ("Severe Non-Proliferative DR", "Many hemorrhages, cotton wool spots, venous beading; high risk of progression.", "stage3.jpg"),
        ("Proliferative DR", "Abnormal new vessels (neovascularization), high risk of bleeding and vision loss.", "stage4.jpg")
    ]

    for stage, desc, img_path in stages:
        with st.expander(f" {stage}"):
            st.write(desc)
            try:
                st.image(img_path, use_container_width=300)
            except:
                st.info(f"Sample image for {stage} would appear here.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Symptoms Guide Tab --------------------
elif tab == "Symptoms Guide":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(" Symptoms Guide")
    st.write("Learn about diabetic retinopathy symptoms and tips for prevention:")

    symptoms = [
        ("Floaters or dark spots in vision", "Small shadows or dark strings appear in your vision due to bleeding in the retina."),
        ("Blurred or distorted vision", "Blood vessel leakage causes distorted vision."),
        ("Poor night vision", "Difficulty seeing at night or in dim light conditions."),
        ("Eye pain or pressure", "Usually in advanced stages, can signal complications."),
        ("Gradual vision loss", "Slow loss of vision over time, often unnoticed initially."),
        ("Difficulty with color perception", "Colors may appear faded or washed out."),
        ("Dark or empty areas in vision", "Caused by bleeding or fluid accumulation in the retina.")
    ]

    for symptom, explanation in symptoms:
        st.markdown(f"<details><summary>{symptom}</summary><p>{explanation}</p></details>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(" Prevention Tips")
    st.write("""
    1. **Control Blood Sugar**: Keep HbA1c below 7%
    2. **Monitor Blood Pressure**: Target <140/90 mmHg
    3. **Regular Eye Exams**: Annual dilated eye exams
    4. **Healthy Diet**: Focus on vegetables, lean proteins, whole grains
    5. **Exercise Regularly**: At least 30 minutes daily
    6. **Quit Smoking**: Smoking increases DR risk significantly
    7. **Manage Cholesterol**: Keep LDL cholesterol in check
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Quiz Tab --------------------
elif tab == "Quiz":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(" Diabetic Retinopathy Quiz")
    st.write("Test your knowledge about diabetic retinopathy!")

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
                    st.success(" Correct!")
                    st.info(q["exp"])
                    st.session_state.quiz_score += 1
                else:
                    st.error(f" Incorrect. Correct: **{q['opts'][q['ans']]}**")
                    st.info(q["exp"])
                st.session_state[f"answered_{i}"] = True

    st.write("---")
    score_percentage = (st.session_state.quiz_score / len(questions)) * 100
    st.write(f"**Your total score: {st.session_state.quiz_score} / {len(questions)} ({score_percentage:.0f}%)**")
    
    if score_percentage >= 80:
        st.success(" Excellent! You have great knowledge about DR!")
    elif score_percentage >= 60:
        st.info(" Good job! Keep learning more about DR.")
    else:
        st.warning(" Consider reviewing the About DR and Symptoms sections.")
    
    if st.button(" Reset Quiz"):
        st.session_state.quiz_score = 0
        for i in range(len(questions)):
            if f"answered_{i}" in st.session_state:
                del st.session_state[f"answered_{i}"]
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Generate Report Tab --------------------
elif tab == "Generate Report":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(" Generate Medical Report")
    
    # Patient Information Form
    st.subheader(" Patient Information")
    
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
    if st.button(" Generate & Download Report", type="primary", use_container_width=300):
        if not patient_name or not patient_age or patient_gender == "Select":
            st.error(" Please fill in all patient information fields.")
        elif not st.session_state.diagnosis_result:
            st.error(" Please complete an AI diagnosis first before generating a report.")
        else:
            # Generate HTML report
            result = st.session_state.diagnosis_result
            
            # Convert images to base64
            img_base64 = ""
            processed_img_base64 = ""
            gradcam_img_base64 = ""
            
            if st.session_state.uploaded_image:
                buffered = BytesIO()
                st.session_state.uploaded_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            if st.session_state.processed_image is not None:
                processed_pil = Image.fromarray(st.session_state.processed_image)
                buffered = BytesIO()
                processed_pil.save(buffered, format="PNG")
                processed_img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            if st.session_state.gradcam_image is not None:
                gradcam_pil = Image.fromarray(st.session_state.gradcam_image)
                buffered = BytesIO()
                gradcam_pil.save(buffered, format="PNG")
                gradcam_img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            severity_colors = {
                "low": "#4CAF50",
                "medium": "#FFC107",
                "high": "#FF9800",
                "critical": "#F44336"
            }
            
            severity_color = severity_colors.get(result.get("severity", "low"), "#2A5C9E")
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        padding: 40px;
                        color: #333;
                        line-height: 1.6;
                    }}
                    .header {{
                        text-align: center;
                        border-bottom: 4px solid {severity_color};
                        padding-bottom: 20px;
                        margin-bottom: 30px;
                    }}
                    .header h1 {{
                        color: {severity_color};
                        margin: 0;
                        font-size: 32px;
                    }}
                    .header p {{
                        color: #666;
                        margin: 10px 0;
                    }}
                    .section {{
                        margin: 25px 0;
                        padding: 20px;
                        background-color: #f9f9f9;
                        border-radius: 8px;
                        border-left: 4px solid {severity_color};
                    }}
                    .section h2 {{
                        color: {severity_color};
                        margin-top: 0;
                        font-size: 24px;
                    }}
                    .info-row {{
                        margin: 12px 0;
                        padding: 8px;
                        background-color: white;
                        border-radius: 4px;
                    }}
                    .label {{
                        font-weight: bold;
                        color: #555;
                        display: inline-block;
                        min-width: 150px;
                    }}
                    .diagnosis-box {{
                        background-color: #E8F5E9;
                        padding: 20px;
                        border-left: 6px solid {severity_color};
                        margin: 20px 0;
                        border-radius: 8px;
                    }}
                    .diagnosis-box h3 {{
                        margin-top: 0;
                        color: {severity_color};
                        font-size: 26px;
                    }}
                    .recommendations {{
                        background-color: white;
                        padding: 15px;
                        border-radius: 6px;
                        margin: 15px 0;
                    }}
                    .recommendations ul {{
                        margin: 10px 0;
                        padding-left: 25px;
                    }}
                    .recommendations li {{
                        margin: 8px 0;
                        line-height: 1.6;
                    }}
                    .image-container {{
                        text-align: center;
                        margin: 20px 0;
                    }}
                    .image-item img {{
                        max-width: 300px;
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        margin: 10px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }}
                    .image-row {{
                        display: flex;
                        justify-content: space-around;
                        flex-wrap: wrap;
                        margin: 20px 0;
                    }}
                    .image-item {{
                        text-align: center;
                        margin: 10px;
                    }}
                    .image-caption {{
                        font-weight: bold;
                        margin-top: 10px;
                        color: #555;
                    }}
                    .footer {{
                        margin-top: 50px;
                        text-align: center;
                        font-size: 12px;
                        color: #777;
                        border-top: 2px solid #ddd;
                        padding-top: 20px;
                    }}
                    .disclaimer {{
                        background-color: #FFF9E6;
                        padding: 15px;
                        border-left: 4px solid #FFC107;
                        margin: 20px 0;
                        border-radius: 6px;
                    }}
                    .severity-badge {{
                        display: inline-block;
                        padding: 8px 16px;
                        background-color: {severity_color};
                        color: white;
                        border-radius: 20px;
                        font-weight: bold;
                        margin: 10px 0;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1> DIABETIC RETINOPATHY DIAGNOSIS REPORT</h1>
                    <p style="font-size: 18px;">AI-Powered Retinal Analysis System</p>
                    <p>Report Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
                </div>
                
                <div class="section">
                    <h2> Patient Information</h2>
                    <div class="info-row"><span class="label">Full Name:</span> {patient_name}</div>
                    <div class="info-row"><span class="label">Age:</span> {patient_age} years</div>
                    <div class="info-row"><span class="label">Gender:</span> {patient_gender}</div>
                    <div class="info-row"><span class="label">Report Date:</span> {datetime.now().strftime("%B %d, %Y")}</div>
                    <div class="info-row"><span class="label">Report Time:</span> {datetime.now().strftime("%I:%M %p")}</div>
                </div>
                
                <div class="section">
                    <h2> AI Diagnosis Results</h2>
                    <div class="diagnosis-box">
                        <h3>Detected Stage: {result['stage']}</h3>
                        <span class="severity-badge">Severity: {result.get('severity', 'N/A').upper()}</span>
                        <div class="info-row"><span class="label">Confidence Level:</span> {result['confidence']*100:.1f}%</div>
                        <div class="info-row"><span class="label">Raw Prediction:</span> {result.get('raw_prediction', 0):.3f} (Scale: 0-4)</div>
                        <div class="info-row" style="margin-top: 15px;">
                            <span class="label">Clinical Findings:</span><br>
                            <p style="margin: 10px 0; padding: 10px; background-color: white; border-radius: 4px;">
                                {result['findings']}
                            </p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2> Recommended Treatment & Management</h2>
                    <div class="recommendations">
                        <ul>
                            {''.join([f'<li>{rec}</li>' for rec in result['recommendations']])}
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <h2> Patient Knowledge Assessment</h2>
                    <div class="info-row"><span class="label">Quiz Score:</span> {st.session_state.quiz_score} / 10</div>
                    <div class="info-row"><span class="label">Percentage:</span> {(st.session_state.quiz_score/10)*100:.0f}%</div>
                    <div class="info-row"><span class="label">Assessment:</span> 
                        {'Excellent understanding of diabetic retinopathy' if st.session_state.quiz_score >= 8 else 
                         'Good understanding, continue learning' if st.session_state.quiz_score >= 6 else 
                         'Needs improvement - patient education recommended'}
                    </div>
                </div>
                
                {"<div class='section'><h2> Retinal Image Analysis</h2>" if img_base64 or processed_img_base64 or gradcam_img_base64 else ""}
                {"<div class='image-row'>" if img_base64 or processed_img_base64 or gradcam_img_base64 else ""}
                    {"<div class='image-item'><img src='data:image/png;base64," + img_base64 + "' alt='Original Image'/><p class='image-caption'>Original Retinal Image</p></div>" if img_base64 else ""}
                    {"<div class='image-item'><img src='data:image/png;base64," + processed_img_base64 + "' alt='Processed Image'/><p class='image-caption'>Preprocessed Image</p></div>" if processed_img_base64 else ""}
                    {"<div class='image-item'><img src='data:image/png;base64," + gradcam_img_base64 + "' alt='GradCAM'/><p class='image-caption'>GradCAM Heatmap</p></div>" if gradcam_img_base64 else ""}
                {"</div></div>" if img_base64 or processed_img_base64 or gradcam_img_base64 else ""}
                
                <div class="disclaimer">
                    <h3 style="margin-top: 0; color: #F57C00;"> Important Disclaimer</h3>
                    <p><strong>This report is generated by an AI system and should NOT replace professional medical advice.</strong></p>
                    <p>The AI model is designed as a screening tool to assist healthcare professionals. All diagnoses should be confirmed by a qualified ophthalmologist through comprehensive clinical examination.</p>
                    <p><strong>Next Steps:</strong></p>
                    <ul>
                        <li>Schedule an appointment with an ophthalmologist for comprehensive eye examination</li>
                        <li>Bring this report to your appointment for reference</li>
                        <li>Continue regular diabetes management and monitoring</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p><strong>AI Model Information:</strong> EfficientNet-B5 Deep Learning Model</p>
                    <p><strong>Training Dataset:</strong> APTOS 2019 Blindness Detection Dataset</p>
                    <p>Report generated by DR Detection System v1.0</p>
                    <p>Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
                    <p style="margin-top: 20px;">For questions or concerns, please consult your healthcare provider.</p>
                </div>
            </body>
            </html>
            """
            
            # Create download button
            st.success(" Report generated successfully!")
            st.download_button(
                label=" Download Report (HTML)",
                data=html_content,
                file_name=f"DR_Report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
            
            st.info(" **Tip:** Open the downloaded HTML file in any web browser to view your complete report. You can print or save it as PDF from your browser.")
            
            # Show preview
            with st.expander(" Preview Report"):
                st.markdown("**Report Summary:**")
                st.write(f"- **Patient:** {patient_name}, {patient_age} years old, {patient_gender}")
                st.write(f"- **Diagnosis:** {result['stage']}")
                st.write(f"- **Confidence:** {result['confidence']*100:.1f}%")
                st.write(f"- **Quiz Score:** {st.session_state.quiz_score}/10")
                st.write(f"- **Images Included:** {'Yes' if img_base64 else 'No'}")
    
    st.markdown('</div>', unsafe_allow_html=True)

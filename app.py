import streamlit as st

# --- CSS Styling for Dark Theme ---
st.markdown("""
    <style>
    body {
        background-color: #0B2545;
        color: #FFFFFF;
    }
    .card {
        background-color: #1B3B70;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2A5C9E;
        color: white;
        border-radius: 8px;
        padding: 0.35em 0.75em;
        margin-top: 5px;
    }
    .stButton>button:hover {
        background-color: #3873C0;
        color: white;
    }
    .stRadio > div {
        flex-direction: column;
    }
    details {
        background-color: #2757A0;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    summary {
        font-weight: bold;
        cursor: pointer;
    }
    img {
        max-width: 100%;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Tabs ---
tabs = ["AI Diagnosis", "About DR", "Symptoms Guide", "Quiz", "Generate Report"]
tab = st.sidebar.radio("Navigation", tabs)

# -------------------- AI Diagnosis Tab --------------------
if tab == "AI Diagnosis":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🤖 AI Diagnosis")
    st.info("This is where your model will predict Diabetic Retinopathy. Coming soon!")
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

    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0

    questions = [
        {"q": "Which part of the eye does diabetic retinopathy primarily affect?", "opts": ["Cornea", "Lens", "Retina", "Optic nerve"], "ans": 2, "exp": "Correct! DR affects retina vessels."},
        {"q": "Which is a common early symptom of diabetic retinopathy?", "opts": ["Sudden blindness", "Floaters or spots", "Eye pain", "Double vision"], "ans": 1, "exp": "Floaters are early signs due to micro-bleeds."},
        {"q": "Which of these increases the likelihood of DR?", "opts": ["Short duration of diabetes", "Poor blood sugar control", "Low blood pressure", "Low cholesterol"], "ans": 1, "exp": "High blood sugar damages retinal vessels."},
        {"q": "How often should a diabetic person get an eye exams?", "opts": ["Every 5 years", "Only if vision declines", "Annually", "Never if no symptoms"], "ans": 2, "exp": "Annual exams catch DR early."},
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
    st.header("📝 Generate Report")
    st.write("This feature will generate a report combining:")
    st.markdown("""
    - Your quiz results  
    - Symptoms overview  
    - AI Diagnosis (when integrated)
    """)
    st.info("Currently placeholder. You can later add PDF export or downloadable report functionality.")
    st.markdown('</div>', unsafe_allow_html=True)

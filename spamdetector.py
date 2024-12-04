import streamlit as st
import pickle

# Load your model and vectorizer
model = pickle.load(open('spam123.pkb1', 'rb'))
cv = pickle.load(open('vec123.pk1', 'rb'))

def main():
    # Sidebar with model info
    st.sidebar.title("Spam Detection App")
    st.sidebar.info("Model developed by **Iswar Kumar Patra** 🧑‍💻")

    # Main page design
    st.markdown(
        """
        <style>
        .main { 
            background-color: #f5f5f5; 
            color: #444; 
            padding: 20px; 
            border-radius: 8px;
        }
        .btn-custom {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 4px;
            border: none;
            cursor: pointer;
        }
        .btn-custom:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("📧 Email Spam Classification Application")
    st.write("Classify your emails as spam or not using a pre-trained machine learning model.")
    st.markdown("---")

    # User input
    user_input = st.text_area("✉️ Enter your email below:", height=200, placeholder="Paste your email content here...")

    # Prediction button with custom style
    if st.button("🔍 Predict", key='predict_button'):
        if user_input.strip():  # Ensure input is not empty
            data = [user_input]
            vec = cv.transform(data).toarray()
            result = model.predict(vec)

            # Display result with different styling
            if result[0] == 0:
                st.success("✅ This is **NOT** a spam email!")
            else:
                st.error("🚨 This **IS** a spam email!")
        else:
            st.warning("⚠️ Please enter some email content!")

    # About section
    st.markdown("---")
    st.write("**Model developed by Iswar Kumar Patra** 🧑‍💻")

if __name__ == '__main__':
    main()

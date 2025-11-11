import streamlit as st
import pickle

# --- 1. Load the Model ---
# Load the trained pipeline model from the pickle file
try:
    with open('spam_classifier_pipeline.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file 'spam_classifier_pipeline.pkl' not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- 2. Streamlit UI ---
st.title('📧 Simple Naive Bayes Email Spam Filter')
st.markdown("""
This app uses a Multinomial Naive Bayes model (trained via a Sklearn Pipeline) 
to classify whether a message is **Ham** (not spam) or **Spam**.
""")

# Text area for user input
email_input = st.text_area(
    "Enter an email message below for classification:",
    "Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!"
)

# Button to trigger classification
if st.button('Classify Message'):
    if email_input:
        # --- 3. Prediction ---
        # The pipeline automatically handles vectorization
        prediction = model.predict([email_input])
        
        # Get prediction result
        result_label = 'SPAM' if prediction[0] == 1 else 'HAM (Not Spam)'
        
        # --- 4. Display Results ---
        st.subheader("Classification Result:")
        
        if prediction[0] == 1:
            st.error(f"🚨 **{result_label}**")
            st.balloons()
        else:
            st.success(f"✅ **{result_label}**")
            
        st.write("---")
        st.caption("A '1' represents Spam, and a '0' represents Ham.")
        st.write(f"Raw Prediction Array: `{prediction}`")
        
    else:
        st.warning("Please enter a message to classify.")
import streamlit as st
import pickle
import pandas as pd  # CRITICAL: Needed for the manual_testing function

# ----------------------------------------------------------------------
# ⚠️ MISSING COMPONENTS: These MUST be defined/loaded for manual_testing to work.
# If you did not save them into my_model.pkl, you need to load them separately.

# Placeholder definitions for external objects (replace with actual loading):
try:
    # 1. Load the fitted Vectorizer
    with open('vectors.pkl', 'rb') as f:
        vectorization = pickle.load(f)
    # 2. Load the trained Logistic Regression model
    with open('lr.pkl', 'rb') as f:
        LR = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Please ensure 'vectorizer.pkl' and 'LR_model.pkl' are in the directory.")
    st.stop()


# 3. Define the wordopt function (must match the training function)
def wordopt(text):
    # This is a placeholder! You must use the EXACT logic 
    # used during your model training (e.g., lowercasing, removing stop words, etc.)
    text = text.lower()
    return text
# ----------------------------------------------------------------------


# 4. Define the output_lable function
def output_lable(n):
    if n==0:
        return "FAKE"
    elif n==1:
        return "REAL"

# 5. Define the manual_testing function (to satisfy the pickle loader)
def manual_testing(news):
    # This function uses the globally loaded 'vectorization', 'LR', and 'wordopt'
    testings_news={"text":[news]}
    new_def_test=pd.DataFrame(testings_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test["text"]
    new_xv_test= vectorization.transform(new_x_test)
    pred_LR=LR.predict(new_xv_test)
    
    # ⚠️ CRITICAL CHANGE: Return the prediction label string, don't just print.
    return output_lable(pred_LR[0])


# ----------------------------------------------------------------------
# 6. Load the pickled object (which should now succeed)
# Since you defined the function, the pickle will load the reference correctly.
try:
    with open('my_model.pkl', 'rb') as file:
        # We are loading the function, but since we already defined it, 
        # this line is mainly to check if the file is okay.
        # However, it's safer to just use the defined manual_testing function directly.
        pass 
except Exception as e:
    st.error(f"Error loading model reference: {e}")
    st.stop()
    
# ----------------------------------------------------------------------

st.title('📰 Fake News Detector')
st.markdown('***')
st.write('Enter a news article or headline to check if it is real or fake.')

# User input
user_input = st.text_area('News Text', height=200, placeholder="Paste the news article text here...")

if st.button('Predict'):
    if user_input.strip() == '':
        st.warning('Please enter some text to analyze.')
    else:
        try:
            # Call the globally defined function
            prediction_label = manual_testing(user_input)
            
            if prediction_label == "FAKE":
                st.error(f'❌ This news is likely **{prediction_label}**.')
            elif prediction_label == "REAL":
                st.success(f'✅ This news is likely **{prediction_label}**.')

        except Exception as e:
            st.error(f"An error occurred during prediction. Check your wordopt/vectorizer setup. Error: {e}")

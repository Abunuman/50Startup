import streamlit as st
import joblib
import numpy as np
# Load the trained model
startup_model = joblib.load('startup_model.pkl')




def main():
    # Streamlit app title
    st.title('Startups Profit Prediction Model ')

    html_temp = """
    <div style="background-color:blue; padding:10px">
    <h3 style="color:white; text-align:center;">An ML model that compare 50 startups by evaluating the profit generated from their R&D, administration, and marketing spend, identifying the most efficient companies that maximize returns from these investments while revealing patterns in resource allocation and profitability</h3>
    </div>

    """

    st.markdown(html_temp, unsafe_allow_html=True)        
    # Input Text for Tv Sales
    rd_input = st.number_input('R&D Spend', min_value=0.0, max_value=1000000.0, value=23.0)
    # Input Text for Radio Sales
    administration_input = st.number_input('Administration', min_value=0.0, max_value=1000000.0, value=23.0)
    # Input Text for Tv Sales
    marketing_input = st.number_input('Marketing Spend', min_value=0.0, max_value=1000000.0, value=20.0)


    # Predict button

    if st.button('Predict'):
        features = np.array([[rd_input, administration_input, marketing_input]])
        prediction = startup_model.predict(features)
    
        st.success(f"Predicted Profit from the Startup: {prediction[0]}")   


        html_temp = """
        <div style="background-color:black; padding:10px"; color:white;>
        <h5 style="color:white; text-align:center;">&copy 2024 Startup Profit Prediction Model trained by: Toyyib Muhammad-Jamiu </h5>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()

# minimal_app.py - SUPER SIMPLE VERSION
import streamlit as st
import requests

st.title("Sentiment Analyzer")

# Text input
text = st.text_input("Enter text:", "I love this product!")

# Analyze button
if st.button("Analyze"):
    try:
        # Call API
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Show result
            if result['sentiment'] == 'positive':
                st.success(f"😊 POSITIVE ({result['confidence']:.2%})")
            else:
                st.error(f"😞 NEGATIVE ({result['confidence']:.2%})")
            
            st.write(f"Text: {result['text']}")
        else:
            st.error(f"API Error: {response.status_code}")
            
    except Exception as e:
        st.error(f"Error: {e}")
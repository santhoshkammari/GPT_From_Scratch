import time

import streamlit as st
import requests

# Define the URL of your FastAPI endpoint
API_URL = "http://localhost:8199/bigram"  # Update with your server's URL

def stream_data(text):
    for line in text.split("\\n"):
        for word in line:
            yield word
            time.sleep(0.03)
        yield '\n'

def main():
    st.title("Bigram Generator")

    user_input = st.text_area("Enter some text:", "")

    if st.button("Generate Bigram", key='generate_button', help='Click to generate bigram'):
        response = requests.post(API_URL, json={"text": user_input})

        if response.status_code == 200:
            st.success("Bigram generated successfully:")
            st.write_stream(stream_data(response.text))
        else:
            st.error("Error generating bigram. Please try again.")


if __name__ == "__main__":
    main()




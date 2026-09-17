import streamlit as st

# streamlit title 
st.title("User Information Form")

# streamlit subheader
st.subheader("Please fill out the form below")

# stramlit text input for name
name = st.text_input("Enter your name")

# streamlit number input for age
age = st.number_input("Please enter your age", min_value=0, max_value=100)

# streamlit selectbox for occupation
occupation = st.selectbox("Select your occupation", ["Student", "Engineer", "Doctor", "Artist", "Other"])

# streamlit radio button for gender
gender = st.radio("Select your gender", ["Male", "Female", "Other"])

# streamlit checkbox for insurance status
insurance_status = st.checkbox("Do you have insurance?")

# streamlit button to submit the form
if st.button("Submit"):
    st.write("You have submitted the form successfully!")
    if name and age and occupation and gender:
        st.write(f"Your Name: **{name}**")
        st.write(f"Your Age: {age}")
        st.write(f"Your Occupation: {occupation}")
        st.write(f"Your Gender: {gender}")
        st.write(f"Insurance Status: {'Yes' if insurance_status else 'No'}")

    



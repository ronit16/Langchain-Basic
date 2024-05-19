import os
from constants import gemini_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import streamlit as st

os.environ["GOOGLE_API_KEY"] = gemini_key
st.title("Celebrity Search")

llm = ChatGoogleGenerativeAI(temperature=0.8, google_api_key=gemini_key, model="gemini-pro")

# first prompt template
first_prompt_template = PromptTemplate(
    input_variables=["name"],
    template="Tell me about {name}"
)

chain1 = LLMChain(
    llm=llm,
    prompt=first_prompt_template,
    verbose=True,
    output_key="details"
)


# second prompt template
second_prompt_template = PromptTemplate(
    input_variables=["details"],
    template="when was {details} born? give me the date in DD-MM-YYYY format"
)

chain2 = LLMChain(
    llm=llm,
    prompt=second_prompt_template,
    verbose=True,
    output_key="DOB"
)

# Third prompt template
Third_prompt_template = PromptTemplate(
    input_variables=["DOB"],
    template="Search for 5 major historical events that happened on {DOB} in any country of the World"
)

chain3 = LLMChain(
    llm=llm,
    prompt=Third_prompt_template,
    verbose=True,
    output_key="Events"
)

parent_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=['name'],
    output_variables=['details', 'DOB', 'Events'], 
    verbose=True
)

input_text = st.text_input("Enter the Celebrity Name")
if input_text:
    result = parent_chain({'name': input_text})
    
    # st expended the text box
    with st.expander("Name"):
        st.write(result['name'])


    with st.expander("Details"):
        st.write(result['details'])


    with st.expander("Date of Birth"):
        st.write(result['DOB'])


    with st.expander("Around this DOB in the world"):
        st.write(result['Events'])
        
    st.write('Data Fetched Successfully')

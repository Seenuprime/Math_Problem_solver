import streamlit as st 
from langchain_groq import ChatGroq
from langchain.chains import llm_math, LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon="||")
st.title("Text to Math Problem Solver Using Google Gemma2")

api = st.sidebar.text_input("Groq API Key:", type="password")

if not api:
    st.info("Please Provide the API Key to access the Model")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", api_key=api)

## Initializing the Tools
wikidia_wrapper = WikipediaAPIWrapper()
wikidia_tool = Tool(
    name="Wikipedia",
    func=wikidia_wrapper.run,
    description="a tool for searching the internet to find the info"
)

## Initialize the Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="tool for answering the math related questions"
)

prompt = """ 
You are agent tasked for solving the given math expressino give me the answer with points structure
expression: {problem}
answer: 
"""

prompt_template = PromptTemplate(input_variables=['question'], template=prompt)

## Math Probelm chain 
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning",
    func=chain.run,
    description="a tool for answering logic based and reasoning questions"
)

## Initialize the agent
assistant_agent = initialize_agent(
    tools = [wikidia_tool, calculator, reasoning_tool],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role": "assistant", "content": "Hi, I'm a Math Chatbot who can answer the given math problems"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

question = st.text_area("Enter your question: ")

if st.button("solve"):
    if question:
        with st.spinner("Generating Response..."):
            st.session_state.messages.append({"role": 'user', "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})

            st.write("## Response:")
            st.write(response)
    
    else:
        st.warning("Please enter the question...")

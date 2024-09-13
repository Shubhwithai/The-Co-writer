import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import os

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Define a simple class to represent a story line
class StoryLine:
    def __init__(self, role, line):
        self.role = role
        self.line = line

# Define a class to manage the story
class Story:
    def __init__(self, story_lines=[]):
        self.story_lines = story_lines

# Set up the Streamlit app
st.header("🧸 The Co-writer")
st.subheader("Write a storyyy!")

# Model selection
model = st.selectbox(
    label="Select the model",
    options=("gpt-4o-mini - OpenAI",),
    index=None,
    placeholder="Select the model!",
)

st.divider()

# Initialize session state for model and LLM
if "model_name" not in st.session_state:
    st.session_state.model_name = ""
if "llm" not in st.session_state:
    st.session_state.llm = ""

# Set the model and LLM based on user selection
if model == "gpt-4o-mini - OpenAI":
    st.session_state.model_name = "gpt-4o-mini"
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)


# Chat input for user line
line = st.chat_input(placeholder="Enter your line")

# Initialize session state for the story
if "story" not in st.session_state:
    st.session_state.story = Story()

# Process user input and generate AI response
if line:
    user_line = StoryLine(role="User", line=line)
    st.session_state.story.story_lines.append(user_line)

    # Construct the prompt for the LLM
    prompt = "\n".join([s.line for s in st.session_state.story.story_lines]) + "\n" + "Predict the next line of the story. Keep it very short."
    
    # Generate AI response
    ai_turn = st.session_state.llm.predict(prompt)
    ai_line = StoryLine(role="AI", line=ai_turn)
    st.session_state.story.story_lines.append(ai_line)

# Display the story
for story_line in st.session_state.story.story_lines:
    st.write(f"{story_line.line}") 
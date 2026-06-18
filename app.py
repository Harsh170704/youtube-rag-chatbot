import streamlit as st
from chatbot_using_RAG import create_chain
def extract_video_id(url):

    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]

    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    elif "/shorts/" in url:
        return url.split("/shorts/")[1].split("?")[0]

    else:
        return None

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥")

st.title("🎥 YouTube RAG Chatbot")

# initialize
if "chain" not in st.session_state:
  st.session_state.chain = None
# youtube url
url = st.text_input(
    "Enter YouTube URL")
# load video button
if st.button("Load Video"):

    try:

        video_id = extract_video_id(url)

        if video_id is None:

            st.error("Invalid YouTube URL")

            st.stop()

        with st.spinner(
            "Loading video and creating RAG pipeline..."
        ):

            st.session_state.chain = create_chain(video_id)

            st.session_state.messages = []

        st.success("Video loaded successfully!")

    except:

        st.error("Invalid YouTube URL")

# ask question

# question = st.text_input("Ask a question about the video")


# if question and st.session_state.chain:
#     with st.spinner("Generating answer..."):
#         answer = (
#             st.session_state
#             .chain
#             .invoke(question)
#         )

#     st.subheader("Answer")

#     st.write(answer)
# initialize chat history

if "messages" not in st.session_state:

    st.session_state.messages = []


# display previous messages

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# chat input at bottom

question = st.chat_input(
    "Ask a question"
)


if question and st.session_state.chain:


    # save user message

    st.session_state.messages.append({

        "role":"user",

        "content":question

    })


    # display user message

    with st.chat_message("user"):

        st.markdown(question)



    # generate answer

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            answer = (
                st.session_state
                .chain
                .invoke(question)
            )

            st.markdown(answer)



    # save assistant answer

    st.session_state.messages.append({

        "role":"assistant",

        "content":answer

    })
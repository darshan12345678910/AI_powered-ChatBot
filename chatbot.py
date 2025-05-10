from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq API client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# User message

def chat_respond(message):
# Create chat completion request
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Give short answers.",
            },
            {
                "role": "user",
                "content": message,
            }
        ],
        model="llama3-8b-8192",  # Use a valid model
        temperature=0.5,
        max_completion_tokens=500,
        top_p=1,
        stop=None,
        stream=False,
    )

    # Print AI response
    res=str(chat_completion.choices[0].message.content)
    return res
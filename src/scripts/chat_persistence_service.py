import os
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
import langchain_openai as openai
from datetime import datetime

load_dotenv(find_dotenv())

MONGO_URI = os.environ.get('MONGO_URI')

client = MongoClient(MONGO_URI)
db = client["KAVAS"]
chat_collection = db["chat-history"]

MAX_TOKEN_COUNT = 2000

def get_token_count(messages):
    """
    Calculate the token count of messages.
    """
    total_token_count = 0
    for msg in messages:
        user_message = msg.get("user_message", "")  # Default to empty string if missing
        ai_message = msg.get("ai_message", "")

        if isinstance(user_message, str) and isinstance(ai_message, str):
            total_token_count += len(user_message.split()) + len(ai_message.split())

    return total_token_count


def summarize_conversation(messages):
    
    token_count = get_token_count(messages)

    if token_count > MAX_TOKEN_COUNT:
        conversation_text = "\n".join([f"User: {msg['user_message']}\nAI: {msg['ai_message']}" for msg in messages])

        try:
            response = openai.Completion.create(
                model="text-davinci-003",  # Use GPT-3 or GPT-4
                prompt=f"Summarize the following conversation:\n{conversation_text}",
                max_tokens=150  # Limit the summary length
            )

            summary = response.choices[0].text.strip()
            return summary
        except Exception as e:
            print("Error summarizing chat history")

            return "Error summarizing conversation"

    else:
        return None


def save_chat_message(user_id, user_message, ai_message):
    timestamp = datetime.utcnow().isoformat()

    user_chat = chat_collection.find_one({"user_id": user_id})

    if user_chat:
        conversation_history = user_chat.get("conversation_history", [])
        conversation_history.append({
            "user_message": user_message,
            "ai_message": ai_message,
            "timestamp": timestamp
        })

        summary = summarize_conversation(conversation_history)

        if summary:
            chat_collection.update_one(
            {"user_id": user_id},
            {"$set": {
                "conversation_summary": summary,
                "last_message": {
                    "user_message": user_message,
                    "ai_message": ai_message,
                    "timestamp": timestamp
                }
            }},
            upsert=True
            )
        else:
            chat_collection.update_one(
                {"user_id": user_id},
                {"$set": {"conversation_history": conversation_history}},
                upsert=True
            )

    else:
        chat_collection.insert_one({
            "user_id": user_id,
            "conversation_history": [{
               "user_message": user_message,
                "ai_message": ai_message,
                "timestamp": timestamp
            }]
        })


def get_chat_history(user_id):
    user_chat = chat_collection.find_one({"user_id": user_id})

    if user_chat:
        if "conversation_summary" in user_chat:
            return {
                "summary": user_chat["conversation_summary"],
                "last_message": user_chat.get("last_message", {})
            }
        else:
            return {"conversation_history": user_chat.get("conversation_history", [])}
    else:
        return None

chat_message = {
    "user":"user1",
    "message": "msg",
    "timestamp": "2025-03-19T12:00:00Z"
}

# chat_collection.insert_one(chat_message)

# for chat in chat_collection.find():
#     print(chat)
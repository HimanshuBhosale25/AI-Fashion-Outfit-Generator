import os
from dotenv import load_dotenv
import openai
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi import Request

# Load environment variables from .env file
load_dotenv()

# API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("CSE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize FastAPI
app = FastAPI()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

class QueryItem(BaseModel):
    query: str
    category: str

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search-item")
async def search_item(query: QueryItem):
    try:
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": query.query,
            "searchType": "image",
            "num": 5
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "items" not in data:
            raise HTTPException(status_code=404, detail="No items found.")

        results = [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
            }
            for item in data["items"]
        ]
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-outfit")
async def generate_outfit(items: list[QueryItem]):
    """
    Generate an image of the final outfit using OpenAI's DALL·E.
    """
    try:
        if not items:
            raise HTTPException(status_code=400, detail="No items provided.")

        # Generate a detailed outfit description using GPT-4o Mini
        outfit_description = generate_gpt4_prompt(items)

        # Log the outfit description for debugging
        print(f"Outfit description: {outfit_description}")

        # Generate the outfit image using DALL·E (OpenAI)
        response = openai.Image.create(
            model="dall-e-3",
            size="1024x1024",
            prompt=outfit_description,
            n=1
        )

        # Log the response for debugging
        print(f"OpenAI response: {response}")

        # Check if the response contains a URL for the generated image
        if 'data' not in response or len(response['data']) == 0:
            raise HTTPException(status_code=500, detail="Error generating image.")

        image_url = response['data'][0]['url']
        return {"image_url": image_url}

    except openai.error.OpenAIError as e:
        # Catch OpenAI specific errors
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")
    except Exception as e:
        # General exception handler
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def generate_gpt4_prompt(items):
    try:
        # Format all the clothing items as a list of descriptions
        item_descriptions = [f"A {item.query} ({item.category})" for item in items]
        item_list = ", ".join(item_descriptions)

        # Use GPT-4o Mini to generate a detailed outfit description
        prompt = f"Create a detailed and fashionable outfit description for a man that includes the following items: {item_list}. " \
                 "The description should include details like color, fabric, style, and how these items fit together. " \
                 "Ensure the outfit looks cohesive and fashionable on a man.Give the final image of a man wearing this outfit."

        # Call GPT-4o Mini to generate the prompt using the correct chat completions endpoint
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Correct model name for GPT-4o Mini
            messages=[{
                "role": "system", "content": "You are a fashion assistant for men."
            }, {
                "role": "user", "content": prompt
            }],
            max_tokens=5000,
            temperature=0.7
        )

        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        print(f"Error during GPT-4o Mini prompt generation: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error generating GPT-4o Mini prompt: {str(e)}")

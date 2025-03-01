# AI Fashion Outfit Generator 👔🕶️

Welcome to the AI Fashion Outfit Generator project! This application helps users discover and create stunning outfits effortlessly using AI technologies. This project focuses on efficient and impactful use of AI to enhance fashion experiences.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [AI Integration](#ai-integration)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)

## Overview
The AI Fashion Outfit Generator is a web application that allows users to search for clothing items, select their favorite pieces, and generate a cohesive outfit image. The application leverages FastAPI for the backend, Google Custom Search API for finding images, and OpenAI's GPT and DALL·E for generating outfit descriptions and images.

## Features ✨
- **Search for Clothing Items:** Users can search for different clothing items using a search bar.
- **Select and Add Items:** Users can add selected items to their outfit list.
- **Generate Outfit:** Using AI, the app generates a detailed outfit description and image.
- **Responsive Design:** The application is designed to work seamlessly on various devices.

## Installation 🛠️
Follow these steps to set up the project locally:

1. **Create and activate a Conda environment:**
    ```bash
    conda create --name fashion-env python=3.10
    conda activate fashion-env
    ```

2. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```
    GOOGLE_API_KEY=your_google_api_key
    CSE_ID=your_cse_id
    OPENAI_API_KEY=your_openai_api_key
    ```

4. **Run the application:**
    ```bash
    uvicorn main:app --reload
    ```

5. **Open your browser:**
    Navigate to `http://127.0.0.1:8000` to use the application.

## Usage 🚀
- Enter a search term in the search bar to find clothing items.
- Select items from the search results to add them to your outfit list.
- Click on "Generate Outfit" to create a detailed outfit description and image.
- The generated outfit image will be displayed on the page.

## AI Integration 🤖
This project utilizes several AI technologies to provide an enhanced user experience:
- **Google Custom Search API:** Fetches images of clothing items based on user queries.
- **OpenAI's GPT:** Generates detailed and fashionable outfit descriptions based on the selected items.
- **OpenAI's DALL·E:** Creates a visual representation of the final outfit.

### How AI is Used
1. **GPT-4o Mini:** This model takes the selected clothing items and generates a detailed and cohesive outfit description. It ensures the outfit looks stylish and includes details like color, fabric, and style. Here's how it works:
    - **Input:** List of selected clothing items.
    - **Process:** Constructs a prompt detailing the outfit, including how the items fit together.
    - **Output:** A descriptive text of the outfit, which is used as a prompt for DALL·E.

2. **DALL·E 3:** This model uses the detailed description generated by GPT-4o Mini to create an image of the final outfit. It interprets the text prompt to generate a realistic visual representation of the outfit.
    - **Input:** Descriptive text of the outfit generated by GPT-4o Mini.
    - **Process:** Utilizes advanced AI to create an image based on the description.
    - **Output:** A high-quality image

## Screenshots 📸
*Home Page:-*
![Home Page](images/Screenshot%202024-12-12%20041829.png)

*Top Wear:-*
![Top Wear](images/Screenshot%202024-12-12%20042121.png)

*Bottom Wear:-*
![Bottom Wear](images/Screenshot%202024-12-12%20042627.png)

*Footwear:-*
![Footwear](images/Screenshot%202024-12-12%20042236.png)

*Selected Options:-*
![Selected Options](images/Screenshot%202024-12-12%20042659.png)

*Generated Outfit 1:-* 
![Generated Outfit1](images/Screenshot%202024-12-12%20042400.png)

*Generated Outfit 2:-*
![Generated Outfit2](images/Screenshot%202024-12-12%20042801.png)

*Generated Outfit 3:-*
![Generated Outfit3](images/Screenshot%202024-12-12%20042847.png)

## Future Improvements 🔮
There is always room for improvement! Here are some potential future enhancements:
- **User Authentication:** Allow users to create accounts and save their favorite outfits.
- **Recommendation System:** Suggest outfits based on user preferences and past selections.
- **Social Sharing:** Enable users to share their generated outfits on social media platforms.
- **Enhanced AI Descriptions:** Utilize more advanced AI models to generate even more detailed and personalized outfit descriptions.
- **Multi-Website Searches:** Currently, the app searches items from Myntra, but it can be extended to support searches from other websites like Amazon, eBay, and more.
- **Price Comparison:** Add functionality to compare prices of the same items across different websites to help users find the best deals.
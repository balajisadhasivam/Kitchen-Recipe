import json
import os

import gradio as gr
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.chains import chain, ingredients_chain, recipe_chain
from src.templates import (
    custom_format_instructions,
    example_instructions,
    first_human_template_str,
    first_system_template_str,
    second_step_template,
)

# Load environment variables
try:
    _ = load_dotenv(find_dotenv())  
    google_api_key = os.environ["GOOGLE_API_KEY"]
except KeyError:
    print("No .env file found, please create it to set GOOGLE_API_KEY environment variable")
    exit(1)  

# Create Chain Definitions
llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
recipe_chain_instance = recipe_chain(llm=llm, system_template=first_system_template_str, human_template=first_human_template_str)
ingredients_chain_instance = ingredients_chain(llm=llm, template=second_step_template)
overall_chain = chain(recipe_chain=recipe_chain_instance, ingredient_chain=ingredients_chain_instance)

# Function to get ingredients
def get_ingredients(food):
    result = overall_chain(
        {
            "food": food,
            "format_instructions": custom_format_instructions,
            "example_instructions": example_instructions,
        }
    )

    recipe = result["recipe"]
    ingredients = result["ingredients"]
    
    #print("Ingredients:", ingredients)  
    
    try:
        dict_ingredients = json.loads(ingredients)
        key_food = list(dict_ingredients.keys())[0]
        output_df = pd.DataFrame(data=dict_ingredients[key_food])
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        output_df = pd.DataFrame()  

    return recipe, output_df
    
# Create Gradio Interface
iface = gr.Interface(
    fn=get_ingredients,
    inputs="text",
    outputs=[gr.outputs.Textbox(label="Recipe"),
        gr.outputs.Dataframe(label="Ingredients"),],
    title="Recipe & Ingredients",
    description="This app provides a food's recipe and the list of ingredients needed to cook it.",
    examples=[["Lasagna"]],
)

if __name__ == "__main__":
    iface.launch()

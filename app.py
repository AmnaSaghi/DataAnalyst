import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent

import matplotlib.pyplot as plt  # Import for plot generation
#from gradio.plots import Plot
import pandas as pd
import io

import os
groq_key = os.getenv('api_key')

def qa_app(csv_file, question):
    df = pd.read_csv(csv_file)
    

    # Update the agent with the new CSV file
    agent = create_csv_agent(ChatGroq(temperature=0, model_name="Llama3-8b-8192",groq_api_key=groq_key),
                             csv_file, verbose=True, return_intermediate_steps=True,handle_parsing_errors=True)

    response = agent.invoke({"input": question})

    ## Check if plot is present in the question
    words = question.lower().split()  # Convert to lowercase and split into words

    plot_code = None
    if "plot" in words:
      fig = plt.figure()
      #print('RESSSSSSSSSSSSSPPP:',response['intermediate_steps'][0][0].tool_input)
      exec(response['intermediate_steps'][0][0].tool_input)
      fig = plt.gcf()
      return response["output"],fig

    else:
      fig = plt.figure()
      return response["output"],fig


# Set up the Gradio interface
demo = gr.Interface(
    fn=qa_app,
    inputs=["file", "text"],
    outputs=["text","plot"],
    #outputs=["text"],
    title="CSV QA App",
    #outputs=["text"],
    #live=False,
    description="Upload a CSV file and ask a question about the data.",
)

# Launch the Gradio app

demo.launch(share=True)

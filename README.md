SPAA Chatbot Test

This repository contains a test Retrieval-Augmented Generation (RAG) chatbot designed for the School of Public Affairs and Administration (SPAA). The chatbot integrates a local Large Language Model (LLM) powered by Ollama, combined with a vector database for document retrieval using LangChain.

You can test it on (https://sites.google.com/view/tcbottest/home)

📌 Project Overview

This is an experimental chatbot designed to answer SPAA-related questions by retrieving information from organizational documents and generating responses using a local LLM. It demonstrates a minimal yet functional RAG pipeline:

Vectorization of documents located in the document/ folder

Query-time retrieval from a vector database

LLM-based answer generation using custom prompts

Simple chat application through main.py

📁 Repository Structure

├── document/ # Folder containing files used to build the vector database

├── vector.py # Vector store and retriever built using LangChain

├── main.py # LLM + prompt + RAG pipeline (Ollama-based chatbot)

└── README.md

Link to Data Source 10.5281/zenodo.18498695 This link contains information scraped from SPAA and OISS website that are used to be indexed into the vector store. These documents serve as the knowledge base for the chatbot.

⚙️ Key Components

local_main.py (last update: 12-22-2025)

Runs the chatbot loop

Loads the Ollama LLM using OllamaLLM

Defines the system prompt

Passes user questions and retrieved context into the model

Implements the RAG chain (prompt → model)

vector.py (last update: 12-22-2025)

Creates or loads a local vector store

Converts documents into embeddings using LangChain embeddings

Builds and exposes a .as_retriever() interface

Used by main.py to retrieve relevant SPAA information during chat

main.py (last update: 04-26-2026)

Web version of 'local_main'

Receive input from 'chatbot-widget.html' and generate output

*12-23-2025 Update: add a new line to let LMM decide and generate search term based on the question. *03-29-2026 Update: added function to detect persona of the users in order to provide acknowledge and tailored responses

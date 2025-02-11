# GraphBot

A project focused on learning and practicing pydantic and AI integration. Built in just a few hours, this serves as an experimental playground for exploring modern development concepts and AI capabilities.

## Work in Progress

This project is actively being developed and is not yet complete. It serves as a learning exercise and a foundation for implementing various features.

### Upcoming Features

- [ ] Chat and Context Management
- [ ] Store images with code metadata in context
- [ ] Properly display observation tables
- [ ] Image Processing Capabilities
- [ ] Table Data Editing Feature
- [ ] User Authentication and Login System

## Project Structure

The project consists of a Python backend using pydantic for data validation and pydantic ai for agentic workflows, and a React frontend for the user interface.


## Running the Project

1. Install Dependencies
   ```bash
   # Install root dependencies
   yarn install

   # Install frontend dependencies
   cd graphbot
   yarn install

   # Install backend dependencies
   cd backend
   pip install -r requirements.txt
   ```

2. Run the Application
   ```bash
   # Run both frontend and backend concurrently
   yarn dev

   # Or run them separately:
   yarn frontend  # Starts React frontend
   yarn backend   # Starts Python backend
   ```

The frontend will run on `http://localhost:3000` and the backend will start on `http://localhost:8000`.
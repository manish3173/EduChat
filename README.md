#  EduChat - Educational Chatbot

An intelligent educational chatbot that helps you learn from PDF documents through interactive conversations, automated summarization, and adaptive quizzing.

## ✨ Features

- 📄 **PDF Document Processing**: Upload and extract text from PDF documents
- 📝 **Intelligent Summarization**: Generate concise summaries using BART-large-CNN model
- 💬 **Interactive Q&A**: Ask questions about document content with RoBERTa-based QA
- ❓ **Adaptive Quiz Generation**: Create quizzes with varying difficulty levels (easy, medium, hard)
- 📊 **Performance Tracking**: Monitor learning progress with detailed analytics
- 🎯 **Multiple Choice Questions**: AI-generated questions with smart distractor options


## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **AI Models**: 
  - `deepset/roberta-base-squad2` for Question Answering
  - `facebook/bart-large-cnn` for Text Summarization
  - `iarfmoose/t5-base-question-generator` for Quiz Generation
- **PDF Processing**: PyPDF2
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib

## 📋 Prerequisites

- Python 3.7+
- pip package manager

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/manish3173/EduChat.git
   cd EduChat
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Run the application**
   ```bash
   streamlit run app.py
   ```



### 1. PDF Upload & Summary
- Upload PDF documents
- Extract and process text content
- Generate intelligent summaries

### 2. Chat with Document
- Ask natural language questions about the document
- Get accurate answers based on document content
- View chat history

### 3. Quiz Generator
- Generate 3-10 questions per quiz
- Choose difficulty level (easy, medium, hard)
- Multiple choice questions with intelligent distractors
- Immediate feedback on answers

### 4. Performance Monitor
- Track overall quiz performance
- View performance trends over time
- Analyze performance by difficulty level
- Detailed quiz history



## Acknowledgments

- Hugging Face for providing excellent transformer models
- Streamlit team for the amazing web app framework
- The open-source community for various libraries used



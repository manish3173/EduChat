import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re
import random
import time
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from typing import List, Dict

# Set page configuration
st.set_page_config(page_title="EduChat - Educational Chatbot", layout="wide")

# Initialize session state variables if they don't exist
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'pdf_summary' not in st.session_state:
    st.session_state.pdf_summary = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []
if 'performance_data' not in st.session_state:
    st.session_state.performance_data = {"quizzes_taken": 0, "correct_answers": 0, "total_questions": 0}
if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = []
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = []

# App title and introduction
st.title("🎓 EduChat - Educational Chatbot")
st.markdown("""
This application helps you learn from PDF documents with the following features:
- 📄 Upload and process PDF documents
- 📝 Generate concise summaries of uploaded content
- 💬 Ask questions about the document content
- ❓ Generate quizzes of varying difficulty levels
- 📊 Track your learning performance
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["PDF Upload & Summary", "Chat with Document", "Quiz Generator", "Performance Monitor"])

# Load models (with caching to prevent reloading)
@st.cache_resource
def load_qa_model():
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return pipeline('question-answering', model=model, tokenizer=tokenizer)

@st.cache_resource
def load_summarizer_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline('summarization', model=model, tokenizer=tokenizer)

@st.cache_resource
def load_qg_model():
    model_name = "iarfmoose/t5-base-question-generator"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline('text2text-generation', model=model, tokenizer=tokenizer)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to generate a summary
def generate_summary(text):
    summarizer = load_summarizer_model()
    # Split text into chunks if it's too long
    max_chunk_length = 1024
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    summaries = []
    for chunk in chunks[:3]:  # Limit to first 3 chunks to avoid overloading
        if len(chunk) > 100:  # Only summarize substantial chunks
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

# Function to answer questions
def answer_question(question, context):
    qa_model = load_qa_model()
    answer = qa_model(question=question, context=context)
    return answer['answer']

# Function to generate quiz questions
class QuizGenerator:
    @staticmethod
    def generate_quiz(
        text: str, 
        num_questions: int = 5, 
        difficulty: str = "easy"
    ) -> List[Dict]:
        """
        Generate a quiz from the given text using a question generation model, with difficulty.
        Each question will have 1 correct answer and 3 distractors (options are shuffled).
        Ensure questions are unique.
        """
        max_attempts = 3
        qa_model = load_qa_model()  # Load once here for efficiency
        for attempt in range(max_attempts):
            questions = generate_questions_from_text(text, num_questions * 3, difficulty)  # Over-generate more to allow deduplication
            # Remove duplicate/empty questions, preserve order, filter out nonsensical questions
            seen = set()
            unique_questions = []
            for q in questions:
                q_clean = q.strip()
                # Improved filtering: exclude questions with too many special chars, or too short, or irrelevant phrases
                if (q_clean and q_clean.lower() not in seen and len(q_clean) > 10 and
                    not re.match(r'^\d+$', q_clean) and
                    not re.match(r'^[\W_]+$', q_clean) and
                    not re.search(r'[^a-zA-Z0-9\s\?\']', q_clean) and
                    all(phrase not in q_clean.lower() for phrase in ["behind locked doors", "46", "true", "false", "n/a", "none of the above", "not sure", "cannot say", "no answer"])):
                    unique_questions.append(q_clean)
                    seen.add(q_clean.lower())
                if len(unique_questions) == num_questions:
                    break
            if len(unique_questions) >= num_questions:
                break
        answers = []
        for question in unique_questions:
            try:
                # Use deep set roberta QA model to generate the correct answer
                answer = qa_model(question=question, context=text)['answer']
                # Improved filtering: exclude answers that are empty, too short, or contain irrelevant phrases
                if (not answer or len(answer.strip()) < 4 or
                    re.match(r'^\d+$', answer.strip()) or
                    re.match(r'^[\W_]+$', answer.strip()) or
                    any(phrase in answer.lower() for phrase in ["behind locked doors", "46", "true", "false", "n/a", "none of the above", "not sure", "cannot say", "no answer"])):
                    answer = "N/A"
            except Exception:
                answer = "N/A"
            answers.append(answer)
        quiz = []
        generic_distractors = ["None of the above", "Not sure", "Cannot say", "No answer"]
        blacklist_phrases = set([phrase.lower() for phrase in generic_distractors] + ["behind locked doors", "n/a"])
        for i, question in enumerate(unique_questions):
            correct_answer = answers[i]
            # Get distractors: pick other answers, ensure uniqueness, not empty, not "N/A", and not blacklisted
            distractors = [a for j, a in enumerate(answers) if j != i and a and a != correct_answer and a != "N/A" and a.lower() not in blacklist_phrases]
            # Prioritize distractors that are meaningful and not generic
            distractors = list(dict.fromkeys(distractors))  # Remove duplicates while preserving order
            # If not enough distractors, use generic distractors without popping destructively
            while len(distractors) < 3:
                distractor = random.choice(generic_distractors)
                if distractor not in distractors and distractor != correct_answer:
                    distractors.append(distractor)
            distractors = distractors[:3]
            options = [correct_answer] + distractors
            random.shuffle(options)
            quiz.append({
                "question": question,
                "options": options,
                "answer": correct_answer
            })
        return quiz

def generate_questions_from_text(text, num_questions=5, difficulty="easy"):
    qg_pipeline = load_qg_model()
    # Split text into manageable chunks (T5 models have a max token limit)
    max_chunk_length = 512
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    questions = []
    for chunk in chunks:
        # The model expects a prefix for question generation, add difficulty
        input_text = f"generate {difficulty} questions: {chunk}"
        outputs = qg_pipeline(input_text, max_length=64, num_return_sequences=1)
        for output in outputs:
            # The model may return multiple questions separated by newlines
            for q in output['generated_text'].split('\n'):
                q = q.strip()
                if q and len(questions) < num_questions:
                    questions.append(q)
        if len(questions) >= num_questions:
            break
    return questions[:num_questions]

# PDF Upload & Summary Page
if page == "PDF Upload & Summary":
    st.header("Upload & Summarize PDF")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from PDF
        with st.spinner('Processing PDF...'):
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
            
        st.success(f"PDF processed successfully! Length: {len(st.session_state.pdf_text)} characters")
        
        if st.button("Generate Summary"):
            with st.spinner('Generating summary...'):
                st.session_state.pdf_summary = generate_summary(st.session_state.pdf_text)
            st.success("Summary generated!")
        
        if st.session_state.pdf_summary:
            st.subheader("Document Summary")
            st.write(st.session_state.pdf_summary)

# Chat with Document Page
elif page == "Chat with Document":
    st.header("Chat with Document")
    
    if not st.session_state.pdf_text:
        st.warning("Please upload a PDF document first.")
    else:
        st.info("Ask questions about the document content.")
        
        # Chat input
        user_question = st.text_input("Your question:")
        
        if user_question:
            with st.spinner('Thinking...'):
                answer = answer_question(user_question, st.session_state.pdf_text)
                
                # Add to chat history
                st.session_state.chat_history.append({"question": user_question, "answer": answer})
            
        # Display chat history
        st.subheader("Chat History")
        for i, exchange in enumerate(st.session_state.chat_history):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"*You:*")
            with col2:
                st.markdown(exchange["question"])
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("*EduChat:*")
            with col2:
                st.markdown(exchange["answer"])
            
            st.divider()
            
        if len(st.session_state.chat_history) == 0:
            st.write("No questions asked yet.")

# Quiz Generator Page
elif page == "Quiz Generator":
    st.header("Quiz Generator")
    
    if not st.session_state.pdf_text:
        st.warning("Please upload a PDF document first.")
    else:
        # Quiz generation form
        with st.form("quiz_generator_form"):
            col1, col2 = st.columns(2)
            with col1:
                num_questions = st.slider("Number of questions", min_value=3, max_value=10, value=5)
            with col2:
                difficulty = st.select_slider("Difficulty level", options=["easy", "medium", "hard"], value="medium")
            
            generate_quiz_button = st.form_submit_button("Generate Quiz")
            
            if generate_quiz_button:
                with st.spinner("Generating quiz questions..."):
                    st.session_state.current_quiz = QuizGenerator.generate_quiz(st.session_state.pdf_text, num_questions, difficulty)
                    st.session_state.quiz_submitted = False
                    st.session_state.user_answers = {}
        
        # Display quiz if available
        if st.session_state.current_quiz:
            if len(st.session_state.current_quiz) > 0 and "options" in st.session_state.current_quiz[0] and len(st.session_state.current_quiz[0]["options"]) > 0:
                with st.form("quiz_form"):
                    for i, q in enumerate(st.session_state.current_quiz):
                        st.markdown(f"*Question {i+1}:* {q['question']}")
                        answer_key = f"q{i}"
                        st.session_state.user_answers[answer_key] = st.radio(
                            f"Select answer for question {i+1}:",
                            q["options"],
                            key=f"quiz_q{i}"
                        )
                        st.divider()
                    
                    submit_quiz = st.form_submit_button("Submit Quiz")
                    
                    if submit_quiz:
                        st.session_state.quiz_submitted = True
                        correct_count = 0
                        results = []
                        
                        for i, q in enumerate(st.session_state.current_quiz):
                            answer_key = f"q{i}"
                            user_answer = st.session_state.user_answers[answer_key]
                            correct = user_answer == q["answer"]
                            if correct:
                                correct_count += 1
                            
                            results.append({
                                "question": q["question"],
                                "user_answer": user_answer,
                                "correct_answer": q["answer"],
                                "is_correct": correct
                            })
                        
                        score_percentage = int((correct_count / len(st.session_state.current_quiz)) * 100)
                        
                        # Update performance data
                        st.session_state.performance_data["quizzes_taken"] += 1
                        st.session_state.performance_data["correct_answers"] += correct_count
                        st.session_state.performance_data["total_questions"] += len(st.session_state.current_quiz)
                        
                        # Save quiz results
                        st.session_state.quiz_results = results
                        st.session_state.quiz_history.append({
                            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                            "difficulty": difficulty,
                            "num_questions": len(st.session_state.current_quiz),
                            "score": score_percentage,
                            "correct": correct_count
                        })
            else:
                st.warning("Could not generate enough quiz questions from the document. Try uploading a more content-rich PDF.")
                
        # Display quiz results if submitted
        if st.session_state.quiz_submitted and hasattr(st.session_state, 'quiz_results'):
            st.subheader("Quiz Results")
            
            correct_count = sum(1 for r in st.session_state.quiz_results if r["is_correct"])
            score_percentage = int((correct_count / len(st.session_state.quiz_results)) * 100)
            
            st.markdown(f"### Your score: {correct_count}/{len(st.session_state.quiz_results)} ({score_percentage}%)")
            
            for i, result in enumerate(st.session_state.quiz_results):
                st.markdown(f"*Question {i+1}:* {result['question']}")
                st.markdown(f"Your answer: {result['user_answer']}")
                st.markdown(f"Correct answer: {result['correct_answer']}")
                
                if result["is_correct"]:
                    st.success("Correct! ✓")
                else:
                    st.error("Incorrect ✗")
                
                st.divider()

# Performance Monitor Page
elif page == "Performance Monitor":
    st.header("Performance Monitor")
    
    if len(st.session_state.quiz_history) == 0:
        st.info("Take some quizzes to see your performance statistics.")
    else:
        # Overall statistics
        st.subheader("Overall Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Quizzes Taken", st.session_state.performance_data["quizzes_taken"])
        
        with col2:
            accuracy = int((st.session_state.performance_data["correct_answers"] / st.session_state.performance_data["total_questions"]) * 100)
            st.metric("Overall Accuracy", f"{accuracy}%")
        
        with col3:
            st.metric("Total Questions Answered", st.session_state.performance_data["total_questions"])
        
        # Convert quiz history to DataFrame for visualization
        history_df = pd.DataFrame(st.session_state.quiz_history)
        
        # Performance over time chart
        st.subheader("Performance Over Time")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(history_df)), history_df['score'], marker='o', linestyle='-')
        ax.set_xlabel('Quiz Number')
        ax.set_ylabel('Score (%)')
        ax.set_xticks(range(len(history_df)))
        ax.set_xticklabels([f"Quiz {i+1}" for i in range(len(history_df))], rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # Performance by difficulty
        st.subheader("Performance by Difficulty")
        difficulty_df = history_df.groupby('difficulty').agg({
            'score': 'mean',
            'num_questions': 'sum',
            'correct': 'sum'
        }).reset_index()
        
        difficulty_df['accuracy'] = (difficulty_df['correct'] / difficulty_df['num_questions']) * 100
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(difficulty_df['difficulty'], difficulty_df['accuracy'], color=['green', 'orange', 'red'])
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}%', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Detailed quiz history
        st.subheader("Quiz History")
        
        # Reformat for better display
        display_df = history_df.copy()
        display_df['performance'] = display_df.apply(
            lambda x: f"{x['correct']}/{x['num_questions']} ({x['score']}%)", axis=1
        )
        display_df = display_df[['date', 'difficulty', 'performance']]
        display_df.columns = ['Date', 'Difficulty', 'Performance']
        
        st.dataframe(display_df, use_container_width=True)

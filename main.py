import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import re
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import wikipedia
from bs4 import BeautifulSoup

# Configure the page
st.set_page_config(
    page_title="AI Learning Roadmap Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "resources" not in st.session_state:
    st.session_state.resources = {}
if "roadmap" not in st.session_state:
    st.session_state.roadmap = []
if "progress" not in st.session_state:
    st.session_state.progress = {}
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""
if "time_limit" not in st.session_state:
    st.session_state.time_limit = 0
if "cache" not in st.session_state:
    st.session_state.cache = {}

# Topic dependency graph - shows prerequisites and learning path
TOPIC_GRAPH = {
    "machine learning": {
        "prerequisites": ["python programming", "mathematics for ai"],
        "next_topics": ["deep learning", "natural language processing", "computer vision"],
        "subtopics": [
            "Introduction to Machine Learning",
            "Supervised Learning",
            "Unsupervised Learning",
            "Feature Engineering",
            "Model Evaluation",
            "Ensemble Methods",
            "Practical Applications"
        ]
    },
    "deep learning": {
        "prerequisites": ["machine learning", "mathematics for ai"],
        "next_topics": ["natural language processing", "computer vision", "reinforcement learning"],
        "subtopics": [
            "Neural Network Fundamentals",
            "Backpropagation",
            "Convolutional Neural Networks",
            "Recurrent Neural Networks",
            "Transformers",
            "Model Optimization",
            "Deep Learning Frameworks"
        ]
    },
    "natural language processing": {
        "prerequisites": ["machine learning", "deep learning"],
        "next_topics": ["large language models", "conversational ai"],
        "subtopics": [
            "Text Preprocessing",
            "Word Embeddings",
            "Language Models",
            "Text Classification",
            "Named Entity Recognition",
            "Machine Translation",
            "Text Generation"
        ]
    },
    "computer vision": {
        "prerequisites": ["machine learning", "deep learning"],
        "next_topics": ["augmented reality", "autonomous systems"],
        "subtopics": [
            "Image Processing",
            "Image Classification",
            "Object Detection",
            "Image Segmentation",
            "Face Recognition",
            "Video Analysis",
            "Generative Models for Images"
        ]
    },
    "reinforcement learning": {
        "prerequisites": ["machine learning", "deep learning"],
        "next_topics": ["game ai", "robotics"],
        "subtopics": [
            "Markov Decision Processes",
            "Q-Learning",
            "Policy Gradients",
            "Deep Q Networks",
            "Actor-Critic Methods",
            "Multi-Agent Systems",
            "Applications in Games and Robotics"
        ]
    },
    "mathematics for ai": {
        "prerequisites": [],
        "next_topics": ["machine learning", "deep learning", "data science"],
        "subtopics": [
            "Linear Algebra",
            "Calculus",
            "Probability Theory",
            "Statistics",
            "Information Theory",
            "Optimization",
            "Numerical Methods"
        ]
    },
    "python programming": {
        "prerequisites": [],
        "next_topics": ["machine learning", "data science", "web development"],
        "subtopics": [
            "Python Basics",
            "Data Structures",
            "Functions and Modules",
            "Object-Oriented Programming",
            "File Handling",
            "Error Handling",
            "Python Libraries for Data Science"
        ]
    },
    "data science": {
        "prerequisites": ["python programming", "mathematics for ai"],
        "next_topics": ["machine learning", "data engineering", "business intelligence"],
        "subtopics": [
            "Data Collection",
            "Data Cleaning",
            "Exploratory Data Analysis",
            "Data Visualization",
            "Statistical Analysis",
            "Feature Engineering",
            "Data Storytelling"
        ]
    },
    "ai ethics": {
        "prerequisites": [],
        "next_topics": ["ai policy", "responsible ai"],
        "subtopics": [
            "Bias and Fairness",
            "Privacy and Security",
            "Transparency and Explainability",
            "Accountability",
            "Social Impact",
            "AI Safety",
            "Ethical Frameworks"
        ]
    },
    "large language models": {
        "prerequisites": ["deep learning", "natural language processing"],
        "next_topics": ["multimodal ai", "ai agents"],
        "subtopics": [
            "Transformer Architecture",
            "Pre-training and Fine-tuning",
            "Prompt Engineering",
            "Retrieval-Augmented Generation",
            "Alignment Techniques",
            "Evaluation Methods",
            "Limitations and Challenges"
        ]
    }
}

# Function to fetch data from Wikipedia
def fetch_wikipedia_data(topic):
    """Fetch information about a topic from Wikipedia"""
    if topic in st.session_state.cache.get('wikipedia', {}):
        return st.session_state.cache['wikipedia'][topic]
    
    try:
        # Search for the topic
        search_results = wikipedia.search(topic, results=5)
        
        if not search_results:
            return {"error": "Topic not found on Wikipedia"}
        
        # Try to get the page for the first search result
        try:
            page = wikipedia.page(search_results[0], auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            # If disambiguation page, use the first option
            page = wikipedia.page(e.options[0], auto_suggest=False)
        
        # Get summary and sections
        summary = page.summary[0:1500]  # Get first 1500 chars of summary
        
        # Get sections (this is a bit tricky with wikipedia package)
        content = page.content
        sections = re.findall(r'==\s*(.*?)\s*==', content)
        
        # Get links
        links = page.links[0:10]  # Get first 10 links
        
        result = {
            "title": page.title,
            "summary": summary,
            "sections": sections,
            "links": links,
            "url": page.url
        }
        
        # Cache the result
        if 'wikipedia' not in st.session_state.cache:
            st.session_state.cache['wikipedia'] = {}
        st.session_state.cache['wikipedia'][topic] = result
        
        return result
    except Exception as e:
        return {"error": f"Error fetching Wikipedia data: {str(e)}"}

# Function to search for educational resources using a free API
def search_educational_resources(topic):
    """Search for educational resources using real APIs"""
    # First check cache
    if topic in st.session_state.cache.get('resources', {}):
        return st.session_state.cache['resources'][topic]
    
    resources = []
    
    # Fetch courses from Coursera API
    try:
        # You'll need to register for a Coursera API key
        coursera_api_key = "YOUR_COURSERA_API_KEY"
        coursera_url = f"https://api.coursera.org/api/courses.v1?q=search&query={topic}&includes=name,description,photoUrl"
        headers = {"Authorization": f"Bearer {coursera_api_key}"}
        
        response = requests.get(coursera_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for course in data.get('elements', [])[:5]:  # Limit to 5 courses
                resources.append({
                    "title": course.get('name', f"{topic} Course"),
                    "type": "course",
                    "platform": "Coursera",
                    "url": f"https://www.coursera.org/learn/{course.get('slug')}",
                    "difficulty": "intermediate",  # You might need to determine this from the course data
                    "duration_hours": 30,  # Estimate or get from API if available
                    "rating": 4.5,  # Get from API if available
                    "quality_score": 0.9  # You might need to calculate this based on ratings
                })
    except Exception as e:
        st.warning(f"Could not fetch courses from Coursera: {str(e)}")
    
    # Fetch YouTube videos
    try:
        youtube_api_key = "AIzaSyCV3LuK5Q6jLNlfgJpJfjBKg_L9cX6k_mQ"
        youtube_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={topic}+tutorial&type=video&key={youtube_api_key}&maxResults=5"
        
        response = requests.get(youtube_url)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('items', []):
                video_id = item['id']['videoId']
                resources.append({
                    "title": item['snippet']['title'],
                    "type": "video",
                    "platform": "YouTube",
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "difficulty": "beginner",  # You might need to estimate this
                    "duration_hours": 1,  # Estimate
                    "rating": 4.0,  # Placeholder
                    "quality_score": 0.8  # Placeholder
                })
    except Exception as e:
        st.warning(f"Could not fetch YouTube videos: {str(e)}")
    
    # Continue with other real APIs...
    
    # Cache the results
    if 'resources' not in st.session_state.cache:
        st.session_state.cache['resources'] = {}
    st.session_state.cache['resources'][topic] = resources
    
    return resources

# Function to assign weights to resources
def assign_weights(resources, user_level="beginner"):
    """Assigns weights to resources based on various factors and user level"""
    weighted_resources = []
    
    # Define importance of different factors based on user level
    if user_level == "beginner":
        difficulty_weights = {"beginner": 1.0, "intermediate": 0.6, "advanced": 0.3}
        type_weights = {"course": 1.0, "tutorial": 0.9, "video": 0.8, "book": 0.7, "article": 0.6}
    elif user_level == "intermediate":
        difficulty_weights = {"beginner": 0.5, "intermediate": 1.0, "advanced": 0.7}
        type_weights = {"course": 0.9, "book": 1.0, "tutorial": 0.8, "video": 0.7, "article": 0.8}
    else:  # advanced
        difficulty_weights = {"beginner": 0.3, "intermediate": 0.7, "advanced": 1.0}
        type_weights = {"article": 0.9, "book": 1.0, "course": 0.8, "tutorial": 0.7, "video": 0.6}
    
    for resource in resources:
        # Base weight from quality score and rating
        base_weight = (resource.get("quality_score", 0.8) * 0.6) + (resource.get("rating", 4.0) / 5.0 * 0.4)
        
        # Adjust weight based on difficulty
        difficulty_factor = difficulty_weights.get(resource.get("difficulty", "intermediate"), 0.7)
        
        # Adjust weight based on resource type
        type_factor = type_weights.get(resource.get("type", "course"), 0.8)
        
        # Calculate final weight
        final_weight = base_weight * difficulty_factor * type_factor
        
        # Add weighted resource to list
        weighted_resource = resource.copy()
        weighted_resource["weight"] = round(final_weight, 2)
        weighted_resources.append(weighted_resource)
    
    # Sort resources by weight (descending)
    weighted_resources.sort(key=lambda x: x["weight"], reverse=True)
    
    return weighted_resources

# Function to get topic subtopics
def get_topic_subtopics(topic):
    """Get subtopics for a given topic"""
    # First check if topic exists in our knowledge graph
    topic_lower = topic.lower()
    
    for known_topic in TOPIC_GRAPH:
        if topic_lower in known_topic or known_topic in topic_lower:
            return TOPIC_GRAPH[known_topic]["subtopics"]
    
    # If not found, get from Wikipedia
    wiki_info = fetch_wikipedia_data(topic)
    if "error" not in wiki_info and "sections" in wiki_info:
        sections = [section for section in wiki_info.get("sections", []) 
                   if not section.lower().startswith(("see also", "references", "external", "notes"))]
        if sections:
            return sections[:7]  # Return up to 7 sections
    
    # If still not found, return generic subtopics
    return [
        f"Introduction to {topic}",
        "Core Concepts",
        "Theoretical Foundations",
        "Practical Applications",
        "Advanced Techniques",
        "Tools and Frameworks",
        "Future Directions"
    ]

# Function to generate a learning roadmap
def generate_roadmap(topic, resources, time_limit=None, user_level="beginner"):
    """Generate a learning roadmap based on the topic, resources, and time limit"""
    
    # Get topic information
    topic_info = fetch_wikipedia_data(topic)
    
    # Get subtopics
    subtopics = get_topic_subtopics(topic)
    
    # Calculate total available time
    total_hours = 0
    if time_limit:
        # Parse time limit string (e.g., "2 weeks", "3 months")
        time_limit = time_limit.lower()
        if "day" in time_limit:
            days = int(re.search(r'(\d+)', time_limit).group(1))
            total_hours = days * 3  # Assuming 3 hours per day
        elif "week" in time_limit:
            weeks = int(re.search(r'(\d+)', time_limit).group(1))
            total_hours = weeks * 15  # Assuming 15 hours per week
        elif "month" in time_limit:
            months = int(re.search(r'(\d+)', time_limit).group(1))
            total_hours = months * 60  # Assuming 60 hours per month
        else:
            try:
                total_hours = int(re.search(r'(\d+)', time_limit).group(1))
            except:
                total_hours = 100  # Default if parsing fails
    else:
        total_hours = 100  # Default if no time limit provided
    
    # Check if topic exists in our knowledge graph
    topic_lower = topic.lower()
    topic_key = None
    for known_topic in TOPIC_GRAPH:
        if topic_lower in known_topic or known_topic in topic_lower:
            topic_key = known_topic
            break
    
    # Get prerequisites if available
    prerequisites = []
    if topic_key and "prerequisites" in TOPIC_GRAPH[topic_key]:
        prerequisites = TOPIC_GRAPH[topic_key]["prerequisites"]
    
    # Get next topics if available
    next_topics = []
    if topic_key and "next_topics" in TOPIC_GRAPH[topic_key]:
        next_topics = TOPIC_GRAPH[topic_key]["next_topics"]
    
    # Sort resources by weight and select top resources
    weighted_resources = assign_weights(resources, user_level)
    selected_resources = []
    
    # Create a map of subtopics to resources
    subtopic_resources = defaultdict(list)
    
    # Determine approximately how many hours to allocate per subtopic
    hours_per_subtopic = total_hours / (len(subtopics) + 1)  # +1 for introduction
    
    # Allocate resources to subtopics
    for subtopic in subtopics:
        subtopic_hours = 0
        for resource in weighted_resources:
            if subtopic_hours >= hours_per_subtopic:
                break
                
            # Check if resource is likely relevant to this subtopic
            resource_title = resource.get("title", "").lower()
            subtopic_lower = subtopic.lower()
            
            # Simple relevance check
            relevant = any(word in resource_title for word in subtopic_lower.split() if len(word) > 3)
            
            # If we couldn't determine relevance, assign based on resource type
            if not relevant:
                # Assign introductory resources to first subtopic
                if subtopic == subtopics[0] and resource.get("difficulty") == "beginner":
                    relevant = True
                # Assign advanced resources to later subtopics
                elif subtopic in subtopics[-3:] and resource.get("difficulty") == "advanced":
                    relevant = True
            
            if relevant and resource not in selected_resources:
                subtopic_resources[subtopic].append(resource)
                selected_resources.append(resource)
                subtopic_hours += resource.get("duration_hours", 0)
    
    # Assign remaining resources to subtopics that have fewer resources
    for resource in weighted_resources:
        if resource not in selected_resources:
            # Find subtopic with fewest resources
            min_resources = float('inf')
            min_subtopic = None
            
            for subtopic in subtopics:
                if len(subtopic_resources[subtopic]) < min_resources:
                    min_resources = len(subtopic_resources[subtopic])
                    min_subtopic = subtopic
            
            if min_subtopic:
                subtopic_resources[min_subtopic].append(resource)
                selected_resources.append(resource)
    
    # Generate the roadmap
    roadmap = []
    start_date = datetime.now()
    current_date = start_date
    
    # Add prerequisites section if available
    if prerequisites:
        prereq_resources = []
        for prereq in prerequisites:
            # Assign next_topic before using it in the dictionary
            next_topic = next_topics[0] if next_topics else "General AI Concepts"
    
            # Add a placeholder resource for each prerequisite
            prereq_resources.append({
                "title": f"Learn {prereq.title()} Fundamentals",
                "type": "prerequisite",
                "platform": "Various",
                "url": f"https://www.google.com/search?q=learn+{next_topic.replace(' ', '+')}",
                "difficulty": "beginner",
                "duration_hours": 10,
                "rating": 4.5,
                "quality_score": 0.85
            })

        
        roadmap.append({
            "phase": "Prerequisites",
            "description": f"Foundation knowledge required before learning {topic}",
            "duration": "1-2 weeks",
            "start_date": current_date.strftime("%Y-%m-%d"),
            "end_date": (current_date + timedelta(days=14)).strftime("%Y-%m-%d"),
            "resources": prereq_resources,
            "completed": False
        })
        
        current_date += timedelta(days=14)
    
    # Add introduction/overview
    intro_resources = [r for r in selected_resources if r.get("difficulty") == "beginner"][:2]
    roadmap.append({
        "phase": "Introduction",
        "description": f"Overview of {topic}",
        "duration": "1 week",
        "start_date": current_date.strftime("%Y-%m-%d"),
        "end_date": (current_date + timedelta(days=7)).strftime("%Y-%m-%d"),
        "resources": intro_resources,
        "completed": False
    })
    
    current_date += timedelta(days=7)
    
    # Add core phases based on subtopics
    for i, subtopic in enumerate(subtopics):
        resources = subtopic_resources[subtopic]
        if not resources:
            continue
            
        # Calculate duration based on resources
        subtopic_hours = sum(resource.get("duration_hours", 0) for resource in resources)
        days = max(3, int(subtopic_hours / 3))  # Assuming 3 hours of study per day
        
        end_date = current_date + timedelta(days=days)
        
        roadmap.append({
            "phase": f"Phase {i+1}: {subtopic}",
            "description": f"Deep dive into {subtopic}",
            "duration": f"{days} days",
            "start_date": current_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "resources": resources,
            "completed": False
        })
        
        current_date = end_date
    
    # Add final project/application phase
    project_resources = [{
        "title": f"Build a {topic} Project",
        "type": "project",
        "platform": "Self-paced",
        "url": "#",
        "difficulty": "intermediate",
        "duration_hours": 20,
        "description": f"Apply your {topic} knowledge by building a practical project"
    }]
    
    roadmap.append({
        "phase": "Final Project",
        "description": f"Apply your {topic} knowledge to a real-world project",
        "duration": "2 weeks",
        "start_date": current_date.strftime("%Y-%m-%d"),
        "end_date": (current_date + timedelta(days=14)).strftime("%Y-%m-%d"),
        "resources": project_resources,
        "completed": False
    })
    
    current_date += timedelta(days=14)
    
    # Add next steps if available
    if next_topics:
        next_resources = []
        for next_topic in next_topics:
            next_resources.append({
                "title": f"Introduction to {next_topic.title()}",
                "type": "next_step",
                "platform": "Various",
                "url": f"https://www.google.com/search?q=learn+{next_topic.replace(' ', '+')}",
                "difficulty": "intermediate",
                "duration_hours": 5,
                "rating": 4.5,
                "quality_score": 0.85
            })
        
        roadmap.append({
            "phase": "Next Steps",
            "description": f"Explore advanced topics after mastering {topic}",
            "duration": "1 week",
            "start_date": current_date.strftime("%Y-%m-%d"),
            "end_date": (current_date + timedelta(days=7)).strftime("%Y-%m-%d"),
            "resources": next_resources,
            "completed": False
        })
    
    return roadmap

# Function to fetch GitHub trends related to a topic
def fetch_github_trends(topic):
    """Fetch trending GitHub repositories related to the topic"""
    try:
        # GitHub API for searching repositories
        github_url = f"https://api.github.com/search/repositories?q={topic}&sort=stars&order=desc"
        
        # GitHub API might require authentication
        #headers = {"Authorization": f"token {your_github_token}"}
        
        response = requests.get(github_url)
        if response.status_code == 200:
            data = response.json()
            trends = []
            for repo in data['items'][:5]:  # Get top 5 repositories
                trends.append({
                    "name": repo['name'],
                    "description": repo['description'] or f"A repository about {topic}",
                    "url": repo['html_url'],
                    "stars": repo['stargazers_count'],
                    "forks": repo['forks_count'],
                    "language": repo['language'] or "Not specified"
                })
            return trends
        else:
            st.warning(f"GitHub API returned status code {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Could not fetch GitHub trends: {str(e)}")
        return []

# Function to generate a learning path visualization
def generate_learning_path(topic):
    """Generate a visual representation of the learning path for a topic"""
    topic_lower = topic.lower()
    
    # Check if topic exists in our knowledge graph
    topic_key = None
    for known_topic in TOPIC_GRAPH:
        if topic_lower in known_topic or known_topic in topic_lower:
            topic_key = known_topic
            break
    
    if not topic_key:
        return None
    
    # Get prerequisites and next topics
    prerequisites = TOPIC_GRAPH[topic_key].get("prerequisites", [])
    next_topics = TOPIC_GRAPH[topic_key].get("next_topics", [])  # Now defined before use
    next_topic = next_topics[0] if next_topics else "AI Basics"  # Safe to use now
    
    # Generate a dictionary representation of the learning path
    learning_path = {
        "topic": topic_key,
        "prerequisites": prerequisites,
        "next_topics": next_topics
    }
    
    return learning_path


# Main Streamlit UI
def main():
    st.title("🧠 AI Learning Roadmap Generator")
    
    with st.sidebar:
        st.header("Configuration")
        topic = st.text_input("Enter a learning topic:", placeholder="e.g., Machine Learning, Python, Deep Learning")
        
        time_options = ["1 week", "2 weeks", "1 month", "3 months", "6 months", "Custom"]
        time_selection = st.selectbox("Learning time frame:", time_options)
        
        if time_selection == "Custom":
            time_limit = st.text_input("Enter custom time frame:", placeholder="e.g., 2 weeks, 3 months")
        else:
            time_limit = time_selection
            
        expertise_level = st.select_slider(
            "Your expertise level:",
            options=["Beginner", "Intermediate", "Advanced"],
            value="Beginner"
        )
            
        if st.button("Generate Roadmap"):
            if topic:
                with st.spinner("Generating your personalized learning roadmap..."):
                    # Update session state
                    st.session_state.current_topic = topic
                    st.session_state.time_limit = time_limit
                    
                    # Fetch resources
                    resources = search_educational_resources(topic)
                    st.session_state.resources = resources
                    
                    # Generate roadmap
                    roadmap = generate_roadmap(
                        topic, 
                        resources, 
                        time_limit, 
                        expertise_level.lower()
                    )
                    st.session_state.roadmap = roadmap
                    
                    # Reset progress
                    st.session_state.progress = {phase["phase"]: False for phase in roadmap}
                    
                    st.success("Roadmap generated! Explore your learning plan.")
            else:
                st.error("Please enter a topic to generate a roadmap.")
        
        if st.session_state.roadmap:
            if st.button("Reset Roadmap"):
                st.session_state.roadmap = []
                st.session_state.resources = {}
                st.session_state.progress = {}
                st.session_state.current_topic = ""
                st.session_state.time_limit = 0
                st.session_state.cache = {}
                st.rerun()
    
    # Main content area
    if not st.session_state.roadmap:
        # Landing page content
        st.markdown("""
        ## Welcome to AI Learning Roadmap Generator!
        
        This tool helps you create a personalized learning plan for any AI or programming topic.
        
        ### How it works:
        1. Enter a topic you'd like to learn
        2. Specify how much time you can dedicate
        3. Choose your expertise level
        4. Get a customized roadmap with resources and timeline
        
        ### Popular topics:
        - Machine Learning
        - Deep Learning
        - Python Programming
        - Data Science
        - Natural Language Processing
        - Computer Vision
        """)
        
        # Display topic suggestions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Machine Learning"):
                st.session_state.current_topic = "Machine Learning"
                st.rerun()
        with col2:
            if st.button("Python Programming"):
                st.session_state.current_topic = "Python Programming"
                st.rerun()
        with col3:
            if st.button("Data Science"):
                st.session_state.current_topic = "Data Science"
                st.rerun()
                
        # Display the knowledge graph
        st.subheader("AI and Programming Learning Path")
        
        # Create a visualization of topic dependencies
        st.write("Below is a simplified learning path for various AI and programming topics:")
        
        # Display topic dependencies
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### Foundation Topics
            - Python Programming
            - Mathematics for AI
            
            ### Intermediate Topics
            - Machine Learning
            - Data Science
            
            ### Advanced Topics
            - Deep Learning
            - Natural Language Processing
            - Computer Vision
            - Reinforcement Learning
            """)
        
        with col2:
            # Display a simple dependency graph
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Hide axes
            ax.axis('off')
            
            # Create a directed graph visualization
            # This is a simplified representation; in a real app, you'd use networkx or similar
            
            # Define node positions
            node_positions = {
                "Python Programming": (0.2, 0.8),
                "Mathematics for AI": (0.8, 0.8),
                "Machine Learning": (0.3, 0.5),
                "Data Science": (0.7, 0.5),
                "Deep Learning": (0.2, 0.3),
                "Natural Language Processing": (0.5, 0.2),
                "Computer Vision": (0.8, 0.3),
                "Reinforcement Learning": (0.5, 0.1)
            }
            
            # Draw nodes
            for node, pos in node_positions.items():
                ax.text(pos[0], pos[1], node, ha='center', va='center', bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="lightblue",
                    ec="steelblue",
                    lw=2
                ))
            
            # Draw edges (arrows)
            arrows = [
                ("Python Programming", "Machine Learning"),
                ("Python Programming", "Data Science"),
                ("Mathematics for AI", "Machine Learning"),
                ("Mathematics for AI", "Data Science"),
                ("Machine Learning", "Deep Learning"),
                ("Machine Learning", "Natural Language Processing"),
                ("Machine Learning", "Computer Vision"),
                ("Deep Learning", "Natural Language Processing"),
                ("Deep Learning", "Computer Vision"),
                ("Deep Learning", "Reinforcement Learning")
            ]
            
            for start, end in arrows:
                start_pos = node_positions[start]
                end_pos = node_positions[end]
                ax.annotate("", 
                            xy=end_pos, 
                            xytext=start_pos,
                            arrowprops=dict(arrowstyle="->", color="gray"))
            
            st.pyplot(fig)
        
    else:
        # Show the generated roadmap
        st.header(f"Learning Roadmap for {st.session_state.current_topic}")
        
        # Display general information about the topic
        with st.expander("Topic Overview", expanded=True):
            wiki_info = fetch_wikipedia_data(st.session_state.current_topic)
            
            if "error" not in wiki_info:
                st.subheader(wiki_info.get("title", st.session_state.current_topic))
                st.write(wiki_info.get("summary", "No summary available"))
                
                # Display related topics
                related_topics = wiki_info.get("links", [])[:5]
                if related_topics:
                    st.write("**Related Topics:**", ", ".join(related_topics))
                    
                # Display source
                st.write(f"Source: [Wikipedia]({wiki_info.get('url', '#')})")
            else:
                st.write(f"Overview for {st.session_state.current_topic}")
        
        # Display learning path visualization
        learning_path = generate_learning_path(st.session_state.current_topic)
        if learning_path:
            with st.expander("Learning Path", expanded=True):
                st.subheader("Topic Dependencies")
                
                # Create columns for prerequisites, current topic, and next topics
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.markdown("### Prerequisites")
                    for prereq in learning_path["prerequisites"]:
                        st.markdown(f"- {prereq.title()}")
                
                with col2:
                    st.markdown("### Current Topic")
                    st.markdown(f"- **{learning_path['topic'].title()}**")
                
                with col3:
                    st.markdown("### Next Topics")
                    for next_topic in learning_path["next_topics"]:
                        st.markdown(f"- {next_topic.title()}")
        
        # Display the roadmap timeline
        st.subheader("Learning Timeline")
        
        # Create a progress bar
        completed_phases = sum(1 for phase in st.session_state.progress.values() if phase)
        total_phases = len(st.session_state.roadmap)
        progress_percentage = int(completed_phases / total_phases * 100) if total_phases > 0 else 0
        
        st.progress(progress_percentage / 100)
        st.write(f"Progress: {progress_percentage}% ({completed_phases}/{total_phases} phases completed)")
        
        # Display each phase of the roadmap
        for i, phase in enumerate(st.session_state.roadmap):
            phase_name = phase["phase"]
            with st.expander(f"{phase_name} ({phase['start_date']} to {phase['end_date']})", expanded=(i == 0)):
                st.write(f"**Description:** {phase['description']}")
                st.write(f"**Duration:** {phase['duration']}")
                
                # Display resources for this phase
                st.write("**Resources:**")
                for resource in phase["resources"]:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"- [{resource['title']}]({resource['url']})")
                    with col2:
                        st.write(f"Type: {resource['type']}")
                    with col3:
                        st.write(f"~{resource['duration_hours']} hours")
                
                # Mark phase as completed
                phase_completed = st.session_state.progress.get(phase_name, False)
                if st.checkbox("Mark as completed", value=phase_completed, key=f"checkbox_{i}"):
                    st.session_state.progress[phase_name] = True
                else:
                    st.session_state.progress[phase_name] = False
        
        # Display trending resources
        with st.expander("Trending Resources", expanded=False):
            st.subheader(f"Trending GitHub Repositories for {st.session_state.current_topic}")
            
            # Fetch trending repositories
            github_trends = fetch_github_trends(st.session_state.current_topic)
            
            if github_trends:
                for repo in github_trends:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"- [{repo['name']}]({repo['url']}): {repo['description']}")
                    with col2:
                        st.write(f"⭐ {repo['stars']}")
                    with col3:
                        st.write(f"Language: {repo['language']}")
            else:
                st.write("No trending repositories found.")
        
        # Display data insights
        with st.expander("Learning Analytics", expanded=False):
            st.subheader("Resource Breakdown")
            
            # Prepare data for visualization
            resource_types = [r["type"] for r in st.session_state.resources]
            type_counts = {}
            for resource_type in resource_types:
                if resource_type in type_counts:
                    type_counts[resource_type] += 1
                else:
                    type_counts[resource_type] = 1
            
            # Create a pie chart of resource types
            fig, ax = plt.subplots()
            ax.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig)
            
            # Show time allocation
            st.subheader("Time Allocation")
            
            # Extract phase durations
            phases = [phase["phase"] for phase in st.session_state.roadmap]
            durations = []
            
            for phase in st.session_state.roadmap:
                total_hours = sum(r.get("duration_hours", 0) for r in phase["resources"])
                durations.append(total_hours)
            
            # Create a bar chart of time allocation
            fig, ax = plt.subplots()
            ax.bar(phases, durations)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()

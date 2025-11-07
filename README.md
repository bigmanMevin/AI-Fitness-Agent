# AI Fitness Recommender with Knowledge Graphs

An intelligent fitness system that combines **Computer Vision**, **Knowledge Graphs**, and **AI** to provide personalized workout recommendations and real-time form correction.

## Features

### 1. **Exercise Tracking with Pose Detection**
- Real-time exercise tracking via webcam
- Video upload for form analysis
- Supports: Squats, Push-ups, Lunges, Plank, and more
- Automatic rep counting
- Live form feedback

### 2. **AI-Powered Form Analysis**
- Uses **Knowledge Graph reasoning** to detect form mistakes
- Provides corrections based on biomechanical principles
- Suggests related exercises to improve weaknesses
- Graph-based progression/regression recommendations

### 3. **Personalized Exercise Recommendations**
- Multi-factor graph reasoning considering:
  - Age and fitness level
  - Injuries and contraindications
  - Fitness goals (weight loss, muscle gain, etc.)
  - Workout history
  - Chat-derived preferences
- Explainable recommendations with graph path visualization

### 4. **AI Chatbot Coach**
- Powered by Ollama (Llama2)
- Context-aware fitness advice
- Integrates with knowledge graph recommendations

## Knowledge Graph Architecture

Built using **NetworkX** with multiple graph layers:

- **Exercise Ontology** - Exercises with attributes (difficulty, muscles, calories)
- **Injury Prevention Graph** - Contraindications with risk scores
- **Form Correction Network** - Mistakes → Corrections → Related Exercises
- **Temporal Graph** - Workout history tracking
- **Similarity Graph** - Exercise clustering via community detection

### Graph Concepts Implemented:
- ✅ Multi-relational directed graphs
- ✅ Weighted edge reasoning
- ✅ Semantic knowledge representation
- ✅ Graph traversal & path finding
- ✅ Constraint satisfaction (injury filtering)
- ✅ Community detection (Louvain algorithm)
- ✅ PageRank for exercise importance
- ✅ Heterogeneous graphs (multiple node types)

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework |
| **MediaPipe** | Pose detection and tracking |
| **OpenCV** | Video/image processing |
| **NetworkX** | Knowledge graph construction and algorithms |
| **Ollama (Llama2)** | Local LLM for chatbot |
| **Plotly & Matplotlib** | Data visualization |
| **NumPy & Pandas** | Data processing |

## Installation

### Prerequisites
- Python 3.8+
- Webcam (for real-time tracking)
- Ollama installed locally

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/AIFitnessRecommender.git
cd AIFitnessRecommender
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama**
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows - Download from https://ollama.com
```

4. **Pull Llama2 model**
```bash
ollama pull llama2
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Open in browser**
```
http://localhost:8501
```

## Usage

### 1. Set Your Profile
- Enter age, height, weight
- Select fitness level (Beginner/Intermediate/Advanced)
- Choose your goal (weight loss, muscle gain, etc.)
- Add any injuries/limitations

### 2. Track Exercises
- **Tab 1: Exercise Tracking**
- Choose webcam or upload video
- Select exercise type
- Start exercising!
- Click "Analyze My Form" for KG-powered corrections

### 3. Get Recommendations
- **Tab 2: KG Recommendations**
- Click "Generate AI Recommendations"
- View personalized exercises based on your profile
- See graph reasoning explanation

### 4. Monitor Progress
- **Tab 3: Progress Analysis**
- Track workout history
- View muscle group coverage
- Identify training gaps

### 5. Chat with AI Coach
- Ask fitness questions at the bottom
- Get personalized advice
- Chat preferences auto-sync with recommendations

## How the Knowledge Graph Works

### Example: User with Knee Pain wanting Weight Loss
```
1. User Profile:
   - Age: 45
   - Injuries: knee_pain
   - Goal: weight_loss

2. Graph Query:
   knee_pain --[contraindicates]--> Squats (risk: 0.8)
   knee_pain --[contraindicates]--> Lunges (risk: 0.9)
   knee_pain --[contraindicates]--> Burpees (risk: 0.7)

3. Safe Exercises:
   weight_loss --[benefits_from]--> Plank (0.7)
   weight_loss --[benefits_from]--> Push-ups (0.6)
   
4. Final Recommendations:
   ✅ Plank (Safe + Effective)
   ✅ Push-ups (Safe + Effective)
   ❌ Squats (Excluded due to injury)
```

## Academic Contributions

This project demonstrates:
- **Semantic Knowledge Representation** in fitness domain
- **Graph-based Reasoning** for personalized recommendations
- **Multi-modal AI** (Vision + Language + Graphs)
- **Explainable AI** through graph path visualization
- **Real-time Integration** of ML models with knowledge graphs

## Future Enhancements

- [ ] Add more exercises (30+ exercises)
- [ ] Implement graph embeddings (Node2Vec)
- [ ] Add nutrition recommendations
- [ ] Multi-user temporal graph analysis
- [ ] Mobile app deployment
- [ ] Link prediction for exercise combinations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- MediaPipe by Google for pose detection
- NetworkX for graph algorithms
- Ollama for local LLM deployment
- Streamlit for the amazing web framework

---

⭐ **If you find this project helpful, please give it a star!** ⭐

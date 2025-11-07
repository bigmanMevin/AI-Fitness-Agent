import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import ollama
import tempfile
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from itertools import combinations

# Page config
st.set_page_config(page_title="AI Fitness KG System", layout="wide")

# --- App Title ---
st.title("AI Fitness System")
st.markdown("""
**Next-Gen Fitness Intelligence** powered by:
- 🕸️ Multi-layered temporal knowledge graphs
- 🤖 Graph neural reasoning for recommendations
- 📊 Community detection for workout clustering
- 🎯 Personalized learning paths using shortest-path algorithms
- 🔄 Real-time graph evolution based on performance
""")

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ============= ADVANCED KNOWLEDGE GRAPH SYSTEM =============

class AdvancedFitnessKG:
    """Multi-layered dynamic knowledge graph with temporal reasoning"""
    
    def __init__(self):
        self.main_graph = nx.DiGraph()
        self.temporal_graph = nx.DiGraph()  # Tracks changes over time
        self.user_graphs = {}  # Individual user graphs
        self.exercise_similarity_graph = nx.Graph()  # Undirected for similarity
        self.injury_prevention_graph = nx.DiGraph()
        self._build_comprehensive_kg()
    
    def _build_comprehensive_kg(self):
        """Build multi-layered knowledge graph"""
        
        # ========== LAYER 1: EXERCISE ONTOLOGY ==========
        exercises = {
            "Squats": {
                "difficulty": 2, "equipment": "none", "calories_per_min": 8,
                "muscles": ["quads", "glutes", "hamstrings", "core"],
                "type": "compound", "plane": "sagittal",
                "skill_level_required": 3, "injury_risk": 2,
                "prerequisites": [], "joint_stress": ["knees", "hips"],
                "biomechanics": "knee_flexion_hip_extension",
                "optimal_knee_angle": (80, 100), "optimal_hip_angle": (70, 90),
                "common_mistakes": ["knees_cave_in", "heels_lift", "back_rounds", "shallow_depth"]
            },
            "Push-ups": {
                "difficulty": 2, "equipment": "none", "calories_per_min": 7,
                "muscles": ["chest", "triceps", "shoulders", "core"],
                "type": "compound", "plane": "transverse",
                "skill_level_required": 3, "injury_risk": 1,
                "prerequisites": ["Plank"], "joint_stress": ["shoulders", "wrists"],
                "biomechanics": "horizontal_push",
                "optimal_elbow_angle": (70, 90), "optimal_body_angle": (160, 180),
                "common_mistakes": ["sagging_hips", "flaring_elbows", "incomplete_range", "head_drops"]
            },
            "Lunges": {
                "difficulty": 2, "equipment": "none", "calories_per_min": 6,
                "muscles": ["quads", "glutes", "hamstrings", "calves"],
                "type": "compound", "plane": "sagittal",
                "skill_level_required": 4, "injury_risk": 3,
                "prerequisites": ["Squats"], "joint_stress": ["knees", "ankles"],
                "biomechanics": "unilateral_knee_flexion",
                "optimal_knee_angle": (80, 100), "optimal_torso_angle": (80, 95),
                "common_mistakes": ["knee_over_toe", "short_stride", "leaning_forward", "wobbly_balance"]
            },
            "Plank": {
                "difficulty": 1, "equipment": "none", "calories_per_min": 5,
                "muscles": ["core", "shoulders", "glutes"],
                "type": "isometric", "plane": "all",
                "skill_level_required": 2, "injury_risk": 1,
                "prerequisites": [], "joint_stress": ["shoulders"],
                "biomechanics": "anti_extension",
                "optimal_body_angle": (165, 185), "optimal_shoulder_angle": (85, 95),
                "common_mistakes": ["sagging_hips", "raised_hips", "head_down", "not_engaging_core"]
            },
            "Jumping Jacks": {
                "difficulty": 1, "equipment": "none", "calories_per_min": 10,
                "muscles": ["full_body", "calves", "shoulders"],
                "type": "cardio", "plane": "frontal",
                "skill_level_required": 2, "injury_risk": 2,
                "prerequisites": [], "joint_stress": ["knees", "ankles"],
                "biomechanics": "plyometric_abduction",
                "common_mistakes": ["bent_knees", "incomplete_arm_raise", "landing_hard"]
            },
            "Burpees": {
                "difficulty": 3, "equipment": "none", "calories_per_min": 12,
                "muscles": ["full_body", "chest", "quads", "shoulders"],
                "type": "cardio", "plane": "all",
                "skill_level_required": 6, "injury_risk": 4,
                "prerequisites": ["Push-ups", "Squats"], "joint_stress": ["knees", "wrists", "shoulders"],
                "biomechanics": "compound_plyometric",
                "common_mistakes": ["poor_pushup_form", "not_jumping", "no_hip_hinge", "rushed_movement"]
            },
            "Mountain Climbers": {
                "difficulty": 2, "equipment": "none", "calories_per_min": 9,
                "muscles": ["core", "shoulders", "quads"],
                "type": "cardio", "plane": "all",
                "skill_level_required": 4, "injury_risk": 2,
                "prerequisites": ["Plank"], "joint_stress": ["shoulders", "wrists"],
                "biomechanics": "dynamic_core_stability",
                "common_mistakes": ["hips_too_high", "slow_pace", "shoulders_forward", "rotation"]
            },
            "Wall Sit": {
                "difficulty": 1, "equipment": "wall", "calories_per_min": 4,
                "muscles": ["quads", "glutes"],
                "type": "isometric", "plane": "sagittal",
                "skill_level_required": 2, "injury_risk": 1,
                "prerequisites": [], "joint_stress": ["knees"],
                "biomechanics": "isometric_knee_flexion",
                "optimal_knee_angle": (85, 95),
                "common_mistakes": ["knees_over_toes", "feet_too_close", "back_not_flat", "uneven_weight"]
            }
        }
        
        # Add exercise nodes with rich attributes
        for ex_name, attrs in exercises.items():
            self.main_graph.add_node(ex_name, node_type="exercise", **attrs)
        
        # ========== NEW: FORM CORRECTION KNOWLEDGE ==========
        # Form mistakes and their corrections
        form_corrections = {
            "knees_cave_in": {
                "issue": "Knees caving inward during squat",
                "causes": ["weak_glutes", "tight_hip_flexors", "poor_ankle_mobility"],
                "corrections": ["Push knees outward", "Engage glutes more", "Widen stance slightly"],
                "cue": "Think about pushing the floor apart with your feet",
                "related_exercises": ["Glute Bridges", "Clamshells", "Wall Sit"],
                "severity": "high"
            },
            "heels_lift": {
                "issue": "Heels lifting off ground during squat",
                "causes": ["tight_calves", "poor_ankle_mobility", "weight_too_forward"],
                "corrections": ["Keep weight on heels", "Improve ankle mobility", "Squat to comfortable depth"],
                "cue": "Imagine you're sitting back into a chair",
                "related_exercises": ["Calf Stretches", "Wall Sit", "Box Squats"],
                "severity": "medium"
            },
            "back_rounds": {
                "issue": "Lower back rounding",
                "causes": ["weak_core", "tight_hamstrings", "going_too_deep"],
                "corrections": ["Keep chest up", "Engage core throughout", "Reduce depth if needed"],
                "cue": "Think proud chest, eyes forward",
                "related_exercises": ["Plank", "Bird Dogs", "Cat-Cow Stretch"],
                "severity": "high"
            },
            "shallow_depth": {
                "issue": "Not squatting deep enough",
                "causes": ["mobility_limitations", "fear_of_depth", "muscle_weakness"],
                "corrections": ["Work on mobility", "Practice with support", "Strengthen legs gradually"],
                "cue": "Aim for thighs parallel to floor",
                "related_exercises": ["Wall Sit", "Goblet Squats", "Hip Mobility Drills"],
                "severity": "low"
            },
            "sagging_hips": {
                "issue": "Hips dropping below shoulder line",
                "causes": ["weak_core", "poor_body_awareness", "fatigue"],
                "corrections": ["Engage core", "Squeeze glutes", "Take breaks when needed"],
                "cue": "Create a straight line from head to heels",
                "related_exercises": ["Forearm Plank", "Dead Bug", "Hollow Body Hold"],
                "severity": "medium"
            },
            "flaring_elbows": {
                "issue": "Elbows pointing out too wide during push-up",
                "causes": ["weak_triceps", "poor_form_habit", "shoulder_mobility"],
                "corrections": ["Keep elbows 45° from body", "Think about pulling shoulder blades together"],
                "cue": "Imagine screwing hands into the floor",
                "related_exercises": ["Tricep Dips", "Close-Grip Push-ups", "Wall Push-ups"],
                "severity": "medium"
            },
            "incomplete_range": {
                "issue": "Not lowering enough in push-up",
                "causes": ["muscle_weakness", "poor_awareness", "rushing"],
                "corrections": ["Lower until chest nearly touches floor", "Control the descent", "Reduce reps if needed"],
                "cue": "Chest should almost kiss the ground",
                "related_exercises": ["Incline Push-ups", "Negative Push-ups", "Plank"],
                "severity": "medium"
            },
            "head_drops": {
                "issue": "Head dropping or looking down",
                "causes": ["neck_fatigue", "poor_posture", "lack_of_focus"],
                "corrections": ["Keep neck neutral", "Look slightly ahead", "Maintain spine alignment"],
                "cue": "Keep a tennis ball between chin and chest",
                "related_exercises": ["Neck Stretches", "Chin Tucks"],
                "severity": "low"
            },
            "knee_over_toe": {
                "issue": "Front knee extending past toes in lunge",
                "causes": ["short_stride", "weight_forward", "poor_balance"],
                "corrections": ["Take a longer step", "Keep shin vertical", "Shift weight to heel"],
                "cue": "Front shin should be perpendicular to floor",
                "related_exercises": ["Split Squats", "Step-Ups", "Wall Sit"],
                "severity": "high"
            },
            "raised_hips": {
                "issue": "Hips raised too high in plank",
                "causes": ["weak_core", "trying_to_make_easier", "poor_awareness"],
                "corrections": ["Lower hips to neutral", "Engage abs", "Tuck pelvis slightly"],
                "cue": "Body should be a straight ramp",
                "related_exercises": ["Modified Plank", "Dead Bug", "Bird Dogs"],
                "severity": "medium"
            }
        }
        
        # Add form correction nodes
        for mistake, attrs in form_corrections.items():
            self.main_graph.add_node(mistake, node_type="form_mistake", **attrs)
        
        # Connect exercises to their common mistakes
        for ex_name, ex_attrs in exercises.items():
            for mistake in ex_attrs.get("common_mistakes", []):
                self.main_graph.add_edge(ex_name, mistake,
                                        relationship="can_have_mistake",
                                        priority=1.0)
        
        # ========== BIOMECHANICAL CUES ==========
        cues = {
            "engage_core": {"description": "Brace your abs as if about to be punched", "applies_to": ["Squats", "Push-ups", "Plank", "Lunges"]},
            "neutral_spine": {"description": "Keep natural curve in lower back", "applies_to": ["Squats", "Push-ups", "Plank"]},
            "chest_up": {"description": "Keep chest proud and shoulders back", "applies_to": ["Squats", "Lunges"]},
            "controlled_tempo": {"description": "2 seconds down, 1 second up", "applies_to": ["Squats", "Push-ups", "Lunges"]},
            "breathe": {"description": "Exhale on exertion, inhale on release", "applies_to": ["Squats", "Push-ups", "Lunges"]},
            "full_range": {"description": "Complete the full movement pattern", "applies_to": ["Squats", "Push-ups", "Lunges"]},
            "stable_base": {"description": "Keep feet firmly planted", "applies_to": ["Squats", "Lunges", "Push-ups"]},
        }
        
        for cue_name, cue_attrs in cues.items():
            self.main_graph.add_node(cue_name, node_type="biomechanical_cue", **cue_attrs)
            
            # Connect cues to exercises
            for exercise in cue_attrs["applies_to"]:
                self.main_graph.add_edge(exercise, cue_name,
                                        relationship="benefits_from_cue")
        
        # ========== REST OF ORIGINAL KG BUILD ==========(continues below)
        
        # ========== REST OF ORIGINAL KG BUILD ==========
        
        # ========== LAYER 2: ANATOMICAL/PHYSIOLOGICAL ENTITIES ==========
        muscle_groups = {
            "quads": {"size": "large", "function": "knee_extension", "recovery_time": 48},
            "glutes": {"size": "large", "function": "hip_extension", "recovery_time": 48},
            "hamstrings": {"size": "large", "function": "knee_flexion", "recovery_time": 48},
            "chest": {"size": "large", "function": "horizontal_push", "recovery_time": 48},
            "shoulders": {"size": "medium", "function": "shoulder_flexion", "recovery_time": 36},
            "triceps": {"size": "small", "function": "elbow_extension", "recovery_time": 24},
            "core": {"size": "medium", "function": "stabilization", "recovery_time": 24},
            "calves": {"size": "small", "function": "ankle_flexion", "recovery_time": 24},
            "full_body": {"size": "all", "function": "systemic", "recovery_time": 72}
        }
        
        for muscle, attrs in muscle_groups.items():
            self.main_graph.add_node(muscle, node_type="muscle_group", **attrs)
        
        # ========== LAYER 3: FITNESS GOALS & ADAPTATIONS ==========
        goals = {
            "weight_loss": {"priority": "caloric_deficit", "training_style": "high_volume"},
            "muscle_gain": {"priority": "progressive_overload", "training_style": "hypertrophy"},
            "endurance": {"priority": "aerobic_capacity", "training_style": "sustained_effort"},
            "strength": {"priority": "max_force", "training_style": "low_rep_high_load"},
            "flexibility": {"priority": "range_of_motion", "training_style": "static_stretch"},
            "athletic_performance": {"priority": "power_speed", "training_style": "plyometric"}
        }
        
        for goal, attrs in goals.items():
            self.main_graph.add_node(goal, node_type="goal", **attrs)
        
        # ========== LAYER 4: BIOMECHANICAL PATTERNS ==========
        movement_patterns = ["push", "pull", "squat", "hinge", "lunge", "carry", "rotation"]
        for pattern in movement_patterns:
            self.main_graph.add_node(pattern, node_type="movement_pattern")
        
        # ========== LAYER 5: INJURY/CONTRAINDICATION NETWORK ==========
        injuries = {
            "knee_pain": {"severity": "medium", "affected_joints": ["knees"]},
            "lower_back_pain": {"severity": "high", "affected_joints": ["spine"]},
            "shoulder_impingement": {"severity": "medium", "affected_joints": ["shoulders"]},
            "wrist_strain": {"severity": "low", "affected_joints": ["wrists"]},
            "ankle_sprain": {"severity": "medium", "affected_joints": ["ankles"]}
        }
        
        for injury, attrs in injuries.items():
            self.injury_prevention_graph.add_node(injury, node_type="injury", **attrs)
        
        # ========== BUILD RELATIONSHIPS ==========
        
        # Exercise -> Muscle relationships (with activation intensity)
        muscle_targeting = {
            "Squats": [("quads", 0.9), ("glutes", 0.8), ("hamstrings", 0.6), ("core", 0.5)],
            "Push-ups": [("chest", 0.9), ("triceps", 0.7), ("shoulders", 0.6), ("core", 0.5)],
            "Lunges": [("quads", 0.8), ("glutes", 0.9), ("hamstrings", 0.7), ("calves", 0.4)],
            "Plank": [("core", 1.0), ("shoulders", 0.5), ("glutes", 0.4)],
            "Jumping Jacks": [("full_body", 0.7), ("calves", 0.6), ("shoulders", 0.5)],
            "Burpees": [("full_body", 0.9), ("chest", 0.7), ("quads", 0.8), ("shoulders", 0.6)],
            "Mountain Climbers": [("core", 0.9), ("shoulders", 0.6), ("quads", 0.5)],
            "Wall Sit": [("quads", 1.0), ("glutes", 0.7)]
        }
        
        for exercise, muscles in muscle_targeting.items():
            for muscle, intensity in muscles:
                self.main_graph.add_edge(exercise, muscle, 
                                        relationship="targets", 
                                        intensity=intensity)
        
        # Goal -> Exercise relationships (effectiveness for goal)
        goal_effectiveness = {
            "weight_loss": [("Burpees", 1.0), ("Jumping Jacks", 0.9), ("Mountain Climbers", 0.85), 
                           ("Squats", 0.7), ("Lunges", 0.7)],
            "muscle_gain": [("Squats", 1.0), ("Push-ups", 1.0), ("Lunges", 0.9), 
                           ("Burpees", 0.7), ("Wall Sit", 0.6)],
            "endurance": [("Burpees", 1.0), ("Mountain Climbers", 0.9), ("Jumping Jacks", 0.85),
                         ("Plank", 0.7)],
            "strength": [("Squats", 0.9), ("Push-ups", 0.9), ("Lunges", 0.8), ("Wall Sit", 0.7)],
            "athletic_performance": [("Burpees", 1.0), ("Jumping Jacks", 0.8), 
                                    ("Mountain Climbers", 0.85), ("Lunges", 0.8)]
        }
        
        for goal, exercises_list in goal_effectiveness.items():
            for exercise, effectiveness in exercises_list:
                self.main_graph.add_edge(goal, exercise, 
                                        relationship="benefits_from",
                                        effectiveness=effectiveness)
        
        # Exercise prerequisites (progression paths)
        prerequisites = {
            "Lunges": ["Squats"],
            "Burpees": ["Push-ups", "Squats"],
            "Mountain Climbers": ["Plank"],
            "Push-ups": ["Plank"]
        }
        
        for advanced_ex, prereq_list in prerequisites.items():
            for prereq_ex in prereq_list:
                self.main_graph.add_edge(prereq_ex, advanced_ex,
                                        relationship="prerequisite_for",
                                        progression_gap=1)
        
        # Injury contraindications
        contraindications = {
            "knee_pain": [("Squats", 0.8), ("Lunges", 0.9), ("Burpees", 0.7), ("Jumping Jacks", 0.6)],
            "lower_back_pain": [("Burpees", 0.7), ("Squats", 0.5), ("Mountain Climbers", 0.4)],
            "shoulder_impingement": [("Push-ups", 0.8), ("Burpees", 0.9), ("Mountain Climbers", 0.6)],
            "wrist_strain": [("Push-ups", 0.7), ("Burpees", 0.8), ("Mountain Climbers", 0.6), ("Plank", 0.5)],
            "ankle_sprain": [("Jumping Jacks", 0.9), ("Lunges", 0.7), ("Burpees", 0.8)]
        }
        
        for injury, ex_list in contraindications.items():
            for exercise, risk_score in ex_list:
                self.injury_prevention_graph.add_edge(injury, exercise,
                                                     relationship="contraindicates",
                                                     risk_score=risk_score)
        
        # ========== BUILD EXERCISE SIMILARITY GRAPH ==========
        self._build_similarity_graph()
    
    def _build_similarity_graph(self):
        """Build exercise similarity graph based on multiple factors"""
        exercises = [n for n in self.main_graph.nodes() 
                    if self.main_graph.nodes[n].get('node_type') == 'exercise']
        
        for ex1, ex2 in combinations(exercises, 2):
            attrs1 = self.main_graph.nodes[ex1]
            attrs2 = self.main_graph.nodes[ex2]
            
            # Calculate similarity score
            similarity = 0
            
            # Muscle overlap
            muscles1 = set(attrs1.get('muscles', []))
            muscles2 = set(attrs2.get('muscles', []))
            muscle_similarity = len(muscles1 & muscles2) / len(muscles1 | muscles2) if muscles1 | muscles2 else 0
            similarity += muscle_similarity * 0.4
            
            # Difficulty similarity
            diff_similarity = 1 - abs(attrs1['difficulty'] - attrs2['difficulty']) / 3
            similarity += diff_similarity * 0.2
            
            # Type similarity
            if attrs1['type'] == attrs2['type']:
                similarity += 0.2
            
            # Biomechanics similarity
            if attrs1.get('biomechanics') == attrs2.get('biomechanics'):
                similarity += 0.2
            
            if similarity > 0.3:  # Only add edge if reasonably similar
                self.exercise_similarity_graph.add_edge(ex1, ex2, weight=similarity)
    
    def get_exercise_communities(self):
        """Detect exercise communities using graph algorithms"""
        from networkx.algorithms import community
        
        # Use Louvain method for community detection
        communities = community.greedy_modularity_communities(self.exercise_similarity_graph)
        
        community_dict = {}
        for i, comm in enumerate(communities):
            community_dict[f"Cluster_{i+1}"] = list(comm)
        
        return community_dict
    
    def calculate_pagerank(self):
        """Calculate exercise importance using PageRank"""
        # Create a version of main graph for PageRank
        pg_graph = self.main_graph.copy()
        
        # Add reverse edges from muscles to exercises (muscle popularity influences exercise)
        for edge in self.main_graph.edges(data=True):
            if edge[2].get('relationship') == 'targets':
                pg_graph.add_edge(edge[1], edge[0], weight=edge[2].get('intensity', 0.5))
        
        try:
            pagerank = nx.pagerank(pg_graph, weight='weight')
            
            # Filter only exercises
            exercise_pagerank = {k: v for k, v in pagerank.items() 
                               if self.main_graph.nodes.get(k, {}).get('node_type') == 'exercise'}
            
            return dict(sorted(exercise_pagerank.items(), key=lambda x: x[1], reverse=True))
        except:
            return {}
    
    def find_optimal_learning_path(self, start_exercise, target_exercise):
        """Find optimal progression path using shortest path algorithms"""
        try:
            path = nx.shortest_path(self.main_graph, start_exercise, target_exercise)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def get_intelligent_recommendations(self, user_profile, workout_history=None):
        """Advanced multi-factor recommendation using graph reasoning"""
        
        fitness_level = user_profile.get('fitness_level', 'Beginner')
        goal = user_profile.get('goal', 'general_fitness')
        injuries = user_profile.get('injuries', [])
        age = user_profile.get('age', 25)
        
        exercise_scores = defaultdict(float)
        exercise_reasons = defaultdict(list)
        
        # Get all exercises
        exercises = [n for n in self.main_graph.nodes() 
                    if self.main_graph.nodes[n].get('node_type') == 'exercise']
        
        for exercise in exercises:
            ex_attrs = self.main_graph.nodes[exercise]
            score = 0
            reasons = []
            
            # CRITICAL: Check injury contraindications FIRST
            is_contraindicated = False
            injury_penalty = 0
            
            for injury in injuries:
                if self.injury_prevention_graph.has_edge(injury, exercise):
                    risk_score = self.injury_prevention_graph.get_edge_data(injury, exercise).get('risk_score', 0)
                    
                    # If high risk (>0.6), exclude completely
                    if risk_score > 0.6:
                        is_contraindicated = True
                        reasons.append(f"⚠️ UNSAFE: High risk for {injury.replace('_', ' ')}")
                        break
                    # Medium risk (0.4-0.6), heavy penalty
                    elif risk_score > 0.4:
                        injury_penalty += risk_score * 0.5
                        reasons.append(f"⚠️ Moderate risk for {injury.replace('_', ' ')}")
                    # Low risk (<0.4), light penalty
                    else:
                        injury_penalty += risk_score * 0.2
                        reasons.append(f"⚠️ Minor concern for {injury.replace('_', ' ')}")
            
            # Skip this exercise if contraindicated
            if is_contraindicated:
                exercise_scores[exercise] = -1.0  # Mark as excluded
                exercise_reasons[exercise] = reasons
                continue
            
            # Factor 1: Goal alignment (30%)
            if self.main_graph.has_edge(goal, exercise):
                effectiveness = self.main_graph.get_edge_data(goal, exercise).get('effectiveness', 0.5)
                score += effectiveness * 0.3
                reasons.append(f"Effective for {goal.replace('_', ' ')}")
            
            # Factor 2: Fitness level appropriateness (25%)
            difficulty = ex_attrs['difficulty']
            level_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
            user_level = level_map.get(fitness_level, 2)
            
            if difficulty <= user_level:
                level_score = 1 - abs(difficulty - user_level) / 3
                score += level_score * 0.25
                reasons.append(f"Matches your {fitness_level} level")
            elif difficulty > user_level + 1:
                # Too difficult
                score -= 0.2
                reasons.append(f"⚠️ May be too difficult")
            
            # Factor 3: Injury penalty (30% weight)
            score -= injury_penalty * 0.3
            
            # Factor 4: Age appropriateness (10%)
            injury_risk = ex_attrs['injury_risk']
            if age > 50 and injury_risk > 2:
                score -= 0.15
                reasons.append("⚠️ High impact for age 50+")
            elif age > 40 and injury_risk > 2:
                score -= 0.1
                reasons.append("Moderate impact for age 40+")
            elif age < 30 and injury_risk < 3:
                score += 0.1
                reasons.append("Age-appropriate intensity")
            
            # Factor 5: Workout history - avoid overtraining same muscles (5%)
            if workout_history:
                recent_muscles = set()
                for hist_ex in workout_history[-3:]:  # Last 3 workouts
                    if self.main_graph.has_node(hist_ex):
                        hist_attrs = self.main_graph.nodes[hist_ex]
                        recent_muscles.update(hist_attrs.get('muscles', []))
                
                # Check muscle overlap
                current_muscles = set(ex_attrs.get('muscles', []))
                overlap = len(current_muscles & recent_muscles)
                if overlap > 2:
                    score -= 0.05 * overlap
                    reasons.append("⚠️ Recently trained muscles")
                elif overlap == 0:
                    score += 0.05
                    reasons.append("Fresh muscle groups")
            
            exercise_scores[exercise] = max(0, score)  # No negative scores (except -1 for excluded)
            exercise_reasons[exercise] = reasons
        
        # Filter out contraindicated exercises (score = -1)
        valid_exercises = {ex: score for ex, score in exercise_scores.items() if score >= 0}
        
        # Sort by score
        sorted_exercises = sorted(valid_exercises.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for ex_name, score in sorted_exercises[:8]:
            ex_data = self.main_graph.nodes[ex_name]
            recommendations.append({
                "name": ex_name,
                "score": score,
                "reasons": exercise_reasons[ex_name],
                "difficulty": ex_data.get("difficulty", 0),
                "calories": ex_data.get("calories_per_min", 0),
                "muscles": ex_data.get("muscles", []),
                "type": ex_data.get("type", ""),
                "injury_risk": ex_data.get("injury_risk", 0)
            })
        
        return recommendations
    
    def update_temporal_graph(self, user_id, exercise, performance_data):
        """Update temporal graph with workout data"""
        timestamp = datetime.now().isoformat()
        node_id = f"{user_id}_{exercise}_{timestamp}"
        
        self.temporal_graph.add_node(node_id,
                                    user_id=user_id,
                                    exercise=exercise,
                                    timestamp=timestamp,
                                    **performance_data)
        
        # Connect to previous workout
        user_nodes = [n for n in self.temporal_graph.nodes() 
                     if self.temporal_graph.nodes[n].get('user_id') == user_id]
        
        if len(user_nodes) > 1:
            # Sort by timestamp and connect to previous
            sorted_nodes = sorted(user_nodes, 
                                key=lambda x: self.temporal_graph.nodes[x].get('timestamp', ''))
            
            if len(sorted_nodes) >= 2:
                self.temporal_graph.add_edge(sorted_nodes[-2], sorted_nodes[-1],
                                            relationship="followed_by")
    
    def analyze_user_progress(self, user_id):
        """Analyze user progress using temporal graph"""
        user_nodes = [n for n in self.temporal_graph.nodes() 
                     if self.temporal_graph.nodes[n].get('user_id') == user_id]
        
        if not user_nodes:
            return None
        
        # Extract data
        progress_data = []
        for node in sorted(user_nodes, key=lambda x: self.temporal_graph.nodes[x].get('timestamp', '')):
            node_data = self.temporal_graph.nodes[node]
            progress_data.append({
                'exercise': node_data.get('exercise'),
                'reps': node_data.get('reps', 0),
                'form_quality': node_data.get('form_quality', 0),
                'timestamp': node_data.get('timestamp')
            })
        
        return progress_data
    
    def analyze_form_and_recommend(self, exercise, detected_issues, angle_data):
        """Analyze form issues and provide intelligent recommendations from KG"""
        
        if not self.main_graph.has_node(exercise):
            return None
        
        ex_data = self.main_graph.nodes[exercise]
        recommendations = {
            "immediate_corrections": [],
            "form_cues": [],
            "related_exercises": [],
            "biomechanical_tips": [],
            "progression_suggestion": None
        }
        
        # Analyze angle data for issues
        detected_mistakes = []
        
        if exercise == "Squats" and angle_data:
            knee_angle = angle_data.get('knee_angle')
            if knee_angle:
                optimal_range = ex_data.get('optimal_knee_angle', (80, 100))
                if knee_angle < optimal_range[0] - 20:
                    detected_mistakes.append("back_rounds")
                elif knee_angle > optimal_range[1] + 40:
                    detected_mistakes.append("shallow_depth")
        
        elif exercise == "Push-ups" and angle_data:
            elbow_angle = angle_data.get('elbow_angle')
            body_angle = angle_data.get('body_angle')
            
            if body_angle and body_angle < 160:
                detected_mistakes.append("sagging_hips")
            if elbow_angle and elbow_angle > 130:
                detected_mistakes.append("incomplete_range")
        
        elif exercise == "Plank" and angle_data:
            body_angle = angle_data.get('body_angle')
            if body_angle:
                if body_angle < 165:
                    detected_mistakes.append("sagging_hips")
                elif body_angle > 185:
                    detected_mistakes.append("raised_hips")
        
        elif exercise == "Lunges" and angle_data:
            knee_angle = angle_data.get('knee_angle')
            if knee_angle and knee_angle > 110:
                detected_mistakes.append("knee_over_toe")
        
        # Get corrections for detected mistakes
        for mistake in detected_mistakes:
            if self.main_graph.has_node(mistake):
                mistake_data = self.main_graph.nodes[mistake]
                
                # Add immediate corrections
                for correction in mistake_data.get('corrections', []):
                    recommendations["immediate_corrections"].append({
                        "issue": mistake_data.get('issue'),
                        "correction": correction,
                        "severity": mistake_data.get('severity'),
                        "cue": mistake_data.get('cue')
                    })
                
                # Add related exercises
                for rel_ex in mistake_data.get('related_exercises', []):
                    if rel_ex not in recommendations["related_exercises"]:
                        recommendations["related_exercises"].append(rel_ex)
        
        # Get biomechanical cues for this exercise
        for neighbor in self.main_graph.neighbors(exercise):
            if self.main_graph.nodes.get(neighbor, {}).get('node_type') == 'biomechanical_cue':
                edge_data = self.main_graph.get_edge_data(exercise, neighbor)
                if edge_data and edge_data.get('relationship') == 'benefits_from_cue':
                    cue_data = self.main_graph.nodes[neighbor]
                    recommendations["form_cues"].append(cue_data.get('description'))
        
        # Progression/regression suggestions based on performance
        if len(detected_mistakes) >= 3:
            # Too many mistakes - suggest regression
            prereqs = []
            for pred in self.main_graph.predecessors(exercise):
                if self.main_graph.nodes.get(pred, {}).get('node_type') == 'exercise':
                    if self.main_graph.get_edge_data(pred, exercise, {}).get('relationship') == 'prerequisite_for':
                        prereqs.append(pred)
            
            if prereqs:
                recommendations["progression_suggestion"] = {
                    "type": "regression",
                    "message": f"Consider mastering {', '.join(prereqs)} first before advancing to {exercise}",
                    "exercises": prereqs
                }
        elif len(detected_mistakes) == 0:
            # Perfect form - suggest progression
            progressions = []
            for succ in self.main_graph.successors(exercise):
                if self.main_graph.nodes.get(succ, {}).get('node_type') == 'exercise':
                    if self.main_graph.get_edge_data(exercise, succ, {}).get('relationship') == 'prerequisite_for':
                        progressions.append(succ)
            
            if progressions:
                recommendations["progression_suggestion"] = {
                    "type": "progression",
                    "message": f"Great form! Ready to try {', '.join(progressions)}?",
                    "exercises": progressions
                }
        
        # Add general biomechanical tips
        recommendations["biomechanical_tips"] = [
            f"Target muscles: {', '.join(ex_data.get('muscles', []))}",
            f"Movement plane: {ex_data.get('plane', 'N/A')}",
            f"Focus on {ex_data.get('biomechanics', 'proper form').replace('_', ' ')}"
        ]
        
        return recommendations

# Initialize KG
@st.cache_resource
def get_fitness_kg():
    return AdvancedFitnessKG()

fitness_kg = get_fitness_kg()

# ============= HELPER FUNCTIONS FOR POSE DETECTION =============

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def get_visibility_score(landmarks, indices):
    """Check if landmarks are visible"""
    return np.mean([landmarks[idx].visibility for idx in indices])

def check_posture(keypoints, exercise_type, landmarks):
    """Check posture for various exercises and return detailed angle data"""
    if not keypoints:
        return "No person detected", False, {}
    
    angle_data = {}
    
    if exercise_type == "Squats":
        visibility = get_visibility_score(landmarks, [
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE
        ])
        
        if visibility < 0.5:
            return "Position yourself sideways", False, {}
        
        knee_angle = calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
        hip_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_hip"], keypoints["left_knee"])
        
        angle_data = {'knee_angle': knee_angle, 'hip_angle': hip_angle}
        
        if knee_angle < 70:
            return "Too deep!", False, angle_data
        elif knee_angle < 90:
            return "Perfect depth! ✓", True, angle_data
        elif knee_angle < 140:
            return "Go deeper", False, angle_data
        else:
            return "Starting position", False, angle_data
    
    elif exercise_type == "Push-ups":
        elbow_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_elbow"], keypoints["left_wrist"])
        body_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_hip"], keypoints["left_ankle"])
        
        angle_data = {'elbow_angle': elbow_angle, 'body_angle': body_angle}
        
        if body_angle < 160:
            return "Keep body straight!", False, angle_data
        
        if elbow_angle < 90:
            return "Perfect depth! ✓", True, angle_data
        elif elbow_angle < 140:
            return "Go deeper", False, angle_data
        else:
            return "Starting position", False, angle_data
    
    elif exercise_type == "Lunges":
        front_knee_angle = calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
        
        angle_data = {'knee_angle': front_knee_angle}
        
        if front_knee_angle < 90:
            return "Perfect lunge! ✓", True, angle_data
        elif front_knee_angle < 130:
            return "Go lower", False, angle_data
        else:
            return "Starting position", False, angle_data
    
    elif exercise_type == "Plank":
        body_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_hip"], keypoints["left_ankle"])
        
        angle_data = {'body_angle': body_angle}
        
        if 165 < body_angle < 185:
            return "Perfect plank! ✓", True, angle_data
        elif body_angle < 165:
            return "Hips too low", False, angle_data
        else:
            return "Hips too high", False, angle_data
    
    return "Exercise not configured", False, angle_data

def count_reps(keypoints, exercise_type, direction, landmarks):
    """Count reps for exercises"""
    rep_complete = False
    new_direction = direction
    angle = None
    
    if exercise_type == "Squats":
        angle = calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
        if angle < 100 and direction == 0:
            new_direction = 1
        elif angle > 140 and direction == 1:
            rep_complete = True
            new_direction = 0
    
    elif exercise_type == "Push-ups":
        angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_elbow"], keypoints["left_wrist"])
        if angle < 100 and direction == 0:
            new_direction = 1
        elif angle > 150 and direction == 1:
            rep_complete = True
            new_direction = 0
    
    elif exercise_type == "Lunges":
        angle = calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
        if angle < 100 and direction == 0:
            new_direction = 1
        elif angle > 140 and direction == 1:
            rep_complete = True
            new_direction = 0
    
    return rep_complete, new_direction, angle

# ============= SIDEBAR =============
st.sidebar.title("🎯 User Profile")

age = st.sidebar.number_input("Age:", 10, 100, 25)
height = st.sidebar.number_input("Height (cm):", 100, 250, 170)
weight = st.sidebar.number_input("Weight (kg):", 30, 200, 70)
fitness_level = st.sidebar.selectbox("Fitness Level:", ["Beginner", "Intermediate", "Advanced"])
fitness_goal = st.sidebar.selectbox("Goal:", ["weight_loss", "muscle_gain", "endurance", 
                                               "strength", "athletic_performance"])

injuries = st.sidebar.multiselect("Current Injuries/Limitations:", 
                                  ["knee_pain", "lower_back_pain", "shoulder_impingement", 
                                   "wrist_strain", "ankle_sprain"])

user_profile = {
    'age': age,
    'height': height,
    'weight': weight,
    'fitness_level': fitness_level,
    'goal': fitness_goal,
    'injuries': injuries
}

st.sidebar.markdown("---")

# Exercise selection
exercise = st.sidebar.selectbox("Current Exercise:", 
                                 ["Squats", "Push-ups", "Lunges", "Plank", 
                                  "Jumping Jacks", "Burpees", "Mountain Climbers", "Wall Sit"])

# Session state
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False
if "rep_count" not in st.session_state:
    st.session_state.rep_count = 0
if "direction" not in st.session_state:
    st.session_state.direction = 0
if "workout_history" not in st.session_state:
    st.session_state.workout_history = []
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{np.random.randint(1000, 9999)}"
if "form_analysis_data" not in st.session_state:
    st.session_state.form_analysis_data = {"angles": [], "issues": []}
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "video_results" not in st.session_state:
    st.session_state.video_results = None
if "chat_preferences" not in st.session_state:
    st.session_state.chat_preferences = {
        "preferred_exercises": [],
        "disliked_exercises": [],
        "additional_goals": [],
        "time_constraints": None,
        "equipment": []
    }

# ============= KNOWLEDGE GRAPH ANALYSIS TABS =============
tab1, tab2, tab3 = st.tabs([
    "🏋️ Exercise Tracking", 
    "🧠 KG Recommendations", 
    "📈 Progress Analysis"
])

# TAB 1: Exercise Tracking
with tab1:
    st.header("Real-Time Exercise Tracking")
    
    # Choose input method
    input_method = st.radio("Choose input method:", ["📹 Webcam", "🎞️ Upload Video"], horizontal=True)
    
    # Initialize uploaded_video variable
    uploaded_video = None
    
    if input_method == "📹 Webcam":
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("▶️ Start Webcam", key="start_btn"):
                st.session_state.webcam_active = True
                st.session_state.rep_count = 0
                st.session_state.direction = 0
        with col2:
            if st.button("⏹️ Stop & Save", key="stop_btn"):
                st.session_state.webcam_active = False
                
                # Save to workout history and temporal graph
                if st.session_state.rep_count > 0:
                    st.session_state.workout_history.append(exercise)
                    
                    # Update temporal graph
                    fitness_kg.update_temporal_graph(
                        st.session_state.user_id,
                        exercise,
                        {
                            'reps': st.session_state.rep_count,
                            'form_quality': 0.85,
                            'duration': 60
                        }
                    )
                    st.success(f"Workout saved! {st.session_state.rep_count} reps of {exercise}")
        
        # Placeholder for video feed
        frame_placeholder = st.empty()
        
        # Real-time webcam processing
        if st.session_state.webcam_active:
            cap = cv2.VideoCapture(0)
            frame_window = st.empty()
            
            while cap.isOpened() and st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam.")
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    keypoints = {
                        "left_hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                        "left_knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                        "left_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
                        "left_shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                        "left_elbow": [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                        "left_wrist": [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
                    }
                    
                    feedback, is_correct, angle_data = check_posture(keypoints, exercise, landmarks)
                    rep_complete, new_direction, _ = count_reps(keypoints, exercise, st.session_state.direction, landmarks)
                    
                    # Store angle data for analysis
                    if angle_data:
                        st.session_state.form_analysis_data["angles"].append(angle_data)
                    
                    if rep_complete:
                        st.session_state.rep_count += 1
                    
                    st.session_state.direction = new_direction
                    
                    color = (0, 255, 0) if is_correct else (255, 165, 0)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2),
                        mp_drawing.DrawingSpec(color=color, thickness=2))
                    
                    cv2.putText(frame, feedback, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
                    if angle_data:
                        # Display primary angle
                        main_angle = angle_data.get('knee_angle') or angle_data.get('elbow_angle') or angle_data.get('body_angle')
                        if main_angle:
                            cv2.putText(frame, f"Angle: {int(main_angle)}°", (10, 90),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Reps: {st.session_state.rep_count}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                frame_window.image(frame, channels="RGB")
            
            cap.release()
            cv2.destroyAllWindows()
    
    else:  # Upload Video
        uploaded_video = st.file_uploader("Upload your workout video:", type=["mp4", "mov", "avi"], key="video_upload")
        
        if uploaded_video:
            # Check if this is a new video
            video_id = f"{uploaded_video.name}_{uploaded_video.size}"
            
            if not st.session_state.video_processed or st.session_state.get("current_video_id") != video_id:
                # New video - process it
                st.session_state.current_video_id = video_id
                st.session_state.video_processed = False
                st.session_state.form_analysis_data = {"angles": [], "issues": []}
                
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                
                st.info("Processing video... This may take a moment.")
                
                cap = cv2.VideoCapture(tfile.name)
                frame_window = st.empty()
                rep_count, direction = 0, 0
                form_scores = []
                
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    current_frame += 1
                    progress_bar.progress(current_frame / total_frames)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)
                    
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        keypoints = {
                            "left_hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                            "left_knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                            "left_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
                            "left_shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                            "left_elbow": [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                            "left_wrist": [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
                        }
                        
                        feedback, is_correct, angle_data = check_posture(keypoints, exercise, landmarks)
                        rep_complete, direction, _ = count_reps(keypoints, exercise, direction, landmarks)
                        
                        # Store angle data
                        if angle_data:
                            st.session_state.form_analysis_data["angles"].append(angle_data)
                        
                        if rep_complete:
                            rep_count += 1
                        
                        if is_correct:
                            form_scores.append(1)
                        else:
                            form_scores.append(0)
                        
                        color = (0, 255, 0) if is_correct else (255, 165, 0)
                        mp_drawing.draw_landmarks(frame_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=color, thickness=2),
                                                  mp_drawing.DrawingSpec(color=color, thickness=2))
                        
                        cv2.putText(frame_rgb, feedback, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 255, 255), 2)
                        if angle_data:
                            main_angle = angle_data.get('knee_angle') or angle_data.get('elbow_angle') or angle_data.get('body_angle')
                            if main_angle:
                                cv2.putText(frame_rgb, f"Angle: {int(main_angle)}°", (10, 90),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.putText(frame_rgb, f"Reps: {rep_count}", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # Show every 5th frame to speed up processing
                    if current_frame % 5 == 0:
                        frame_window.image(frame_rgb)
                
                cap.release()
                progress_bar.empty()
                frame_window.empty()
                
                # Store results
                form_quality = (sum(form_scores) / len(form_scores) * 100) if form_scores else 0
                st.session_state.video_results = {
                    'rep_count': rep_count,
                    'form_quality': form_quality,
                    'total_frames': total_frames
                }
                st.session_state.video_processed = True
                st.session_state.rep_count = rep_count
                
                st.success(f"✅ Video processing complete!")
            
            # Display stored results
            if st.session_state.video_processed and st.session_state.video_results:
                results = st.session_state.video_results
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reps", results['rep_count'])
                with col2:
                    st.metric("Form Quality", f"{results['form_quality']:.0f}%")
                with col3:
                    calories = results['rep_count'] * 0.5
                    st.metric("Est. Calories", f"{calories:.0f}")
                
                # Save to history
                if st.button("💾 Save to Workout History", key="save_video_workout"):
                    st.session_state.workout_history.append(exercise)
                    
                    # Update temporal graph
                    fitness_kg.update_temporal_graph(
                        st.session_state.user_id,
                        exercise,
                        {
                            'reps': results['rep_count'],
                            'form_quality': results['form_quality'] / 100,
                            'duration': results['total_frames'] / 30
                        }
                    )
                    st.success(f"Saved! {results['rep_count']} reps of {exercise} added to your history")
    
    # Exercise metrics (for webcam)
    if input_method == "📹 Webcam" and st.session_state.rep_count > 0:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Reps", st.session_state.rep_count)
        with col2:
            st.metric("Form Score", "85%")
        with col3:
            calories = st.session_state.rep_count * 0.5
            st.metric("Calories", f"{calories:.0f}")
    
    # ========== FORM ANALYSIS FROM KNOWLEDGE GRAPH ==========
    st.markdown("---")
    st.subheader("🧠 AI Form Analysis (Knowledge Graph)")
    
    # Check if we have data to analyze
    has_data = (st.session_state.rep_count > 0 and input_method == "📹 Webcam") or \
               (uploaded_video is not None and input_method == "🎞️ Upload Video")
    
    if has_data and st.session_state.form_analysis_data["angles"]:
        if st.button("🔍 Analyze My Form", key="analyze_form_btn"):
            with st.spinner("Analyzing your form using Knowledge Graph..."):
                # Get average angle data
                avg_angles = {}
                angle_keys = st.session_state.form_analysis_data["angles"][0].keys()
                for key in angle_keys:
                    values = [d[key] for d in st.session_state.form_analysis_data["angles"] if key in d]
                    if values:
                        avg_angles[key] = sum(values) / len(values)
                
                # Get recommendations from KG
                recommendations = fitness_kg.analyze_form_and_recommend(
                    exercise,
                    st.session_state.form_analysis_data.get("issues", []),
                    avg_angles
                )
                
                if recommendations:
                    # Display immediate corrections
                    if recommendations["immediate_corrections"]:
                        st.markdown("### ⚠️ Form Issues Detected")
                        for i, correction in enumerate(recommendations["immediate_corrections"], 1):
                            severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                            severity_icon = severity_color.get(correction["severity"], "🔵")
                            
                            with st.expander(f"{severity_icon} {correction['issue']}", expanded=True):
                                st.markdown(f"**Fix:** {correction['correction']}")
                                st.markdown(f"**Cue:** *\"{correction['cue']}\"*")
                    else:
                        st.success("✅ Excellent form! No major issues detected.")
                    
                    # Display form cues
                    if recommendations["form_cues"]:
                        st.markdown("### 💡 Key Form Cues")
                        for cue in recommendations["form_cues"]:
                            st.info(f"✓ {cue}")
                    
                    # Display biomechanical tips
                    if recommendations["biomechanical_tips"]:
                        st.markdown("### 🔬 Biomechanical Insights")
                        for tip in recommendations["biomechanical_tips"]:
                            st.markdown(f"- {tip}")
                    
                    # Display related exercises
                    if recommendations["related_exercises"]:
                        st.markdown("### 🏋️ Supplementary Exercises")
                        st.markdown("To improve your form, try these exercises:")
                        cols = st.columns(min(len(recommendations["related_exercises"]), 4))
                        for idx, rel_ex in enumerate(recommendations["related_exercises"]):
                            with cols[idx % 4]:
                                st.button(rel_ex, key=f"rel_ex_{idx}", use_container_width=True)
                    
                    # Display progression/regression suggestion
                    if recommendations["progression_suggestion"]:
                        suggestion = recommendations["progression_suggestion"]
                        if suggestion["type"] == "progression":
                            st.success(f"🎯 {suggestion['message']}")
                        else:
                            st.warning(f"📉 {suggestion['message']}")
                        
                        st.markdown("**Suggested exercises:**")
                        for ex in suggestion["exercises"]:
                            st.markdown(f"- {ex}")
    elif has_data and not st.session_state.form_analysis_data["angles"]:
        st.info("⏳ Complete your workout to collect form data for analysis!")
    else:
        st.info("📹 Start exercising to unlock AI form analysis!")

# TAB 2: KG Recommendations
with tab2:
    st.header("🧠 Graph-Based Intelligent Recommendations")
    
    # Add chat-enhanced toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Get recommendations based on your profile **+ Chat preferences**")
    with col2:
        use_chat_context = st.toggle("Use Chat Context", value=True, help="Include preferences from your chat conversations")
    
    # Show current profile
    st.subheader("📋 Your Profile")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Age", f"{age} years")
    with col2:
        st.metric("Level", fitness_level)
    with col3:
        st.metric("Goal", fitness_goal.replace('_', ' ').title())
    with col4:
        injury_count = len(injuries) if injuries else 0
        st.metric("Injuries", injury_count)
    
    if injuries:
        st.warning(f"⚠️ Active Injuries: {', '.join([i.replace('_', ' ').title() for i in injuries])}")
    
    # Show chat-derived preferences
    if use_chat_context and st.session_state.chat_preferences:
        prefs = st.session_state.chat_preferences
        has_prefs = any([
            prefs['preferred_exercises'],
            prefs['disliked_exercises'],
            prefs['additional_goals'],
            prefs['time_constraints'],
            prefs['equipment']
        ])
        
        if has_prefs:
            with st.expander("💬 Preferences from Your Chat History", expanded=True):
                if prefs['preferred_exercises']:
                    st.success(f"✅ You like: {', '.join(prefs['preferred_exercises'])}")
                if prefs['disliked_exercises']:
                    st.warning(f"❌ You dislike: {', '.join(prefs['disliked_exercises'])}")
                if prefs['additional_goals']:
                    st.info(f"🎯 Additional goals: {', '.join(prefs['additional_goals'])}")
                if prefs['time_constraints']:
                    st.info(f"⏰ Time available: {prefs['time_constraints']}")
                if prefs['equipment']:
                    st.info(f"🏋️ Equipment: {', '.join(prefs['equipment'])}")
    
    st.markdown("---")
    
    if st.button("🔍 Generate AI Recommendations", key="rec_btn", type="primary", use_container_width=True):
        with st.spinner("Analyzing knowledge graph with your profile + chat context..."):
            # Get base recommendations
            recommendations = fitness_kg.get_intelligent_recommendations(
                user_profile,
                st.session_state.workout_history
            )
            
            # Apply chat context if enabled
            if use_chat_context and st.session_state.chat_preferences:
                prefs = st.session_state.chat_preferences
                
                # Boost preferred exercises
                for rec in recommendations:
                    if rec['name'] in prefs['preferred_exercises']:
                        rec['score'] *= 1.3
                        rec['reasons'].insert(0, "💬 You mentioned you like this exercise")
                    
                    # Penalize disliked exercises
                    if rec['name'] in prefs['disliked_exercises']:
                        rec['score'] *= 0.3
                        rec['reasons'].insert(0, "💬 You mentioned you dislike this - showing alternatives")
                    
                    # Boost if matches additional goals
                    for goal in prefs['additional_goals']:
                        if goal in rec['type'] or goal in ' '.join(rec['muscles']):
                            rec['score'] *= 1.15
                            rec['reasons'].insert(0, f"💬 Matches your goal: {goal}")
                
                # Re-sort after applying preferences
                recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        st.success("✅ Recommendations generated using graph reasoning + chat context!")
        
        # Show reasoning process
        with st.expander("🧠 How the Knowledge Graph Analyzed Your Profile", expanded=False):
            st.markdown(f"""
            **Graph Traversal Process:**
            
            1️⃣ **User Node Properties:**
            - Age: {age} → {"Low injury risk exercises" if age > 40 else "Can handle higher intensity"}
            - Fitness Level: {fitness_level} → Filtered exercises by difficulty
            - Goal: {fitness_goal} → Traversed goal→exercise edges with effectiveness weights
            """)
            
            if use_chat_context and st.session_state.chat_preferences:
                prefs = st.session_state.chat_preferences
                if any([prefs['preferred_exercises'], prefs['disliked_exercises']]):
                    st.markdown("""
            
            💬 **Chat Context Applied:**
            """)
                    if prefs['preferred_exercises']:
                        st.markdown(f"- Boosted scores (+30%): {', '.join(prefs['preferred_exercises'])}")
                    if prefs['disliked_exercises']:
                        st.markdown(f"- Reduced scores (-70%): {', '.join(prefs['disliked_exercises'])}")
            
            st.markdown(f"""
            2️⃣ **Injury Constraint Graph:**
            """)
            
            if injuries:
                st.markdown("⚠️ **Active contraindications found:**")
                for injury in injuries:
                    contraindicated = []
                    if fitness_kg.injury_prevention_graph.has_node(injury):
                        for neighbor in fitness_kg.injury_prevention_graph.neighbors(injury):
                            edge_data = fitness_kg.injury_prevention_graph.get_edge_data(injury, neighbor)
                            if edge_data and edge_data.get('relationship') == 'contraindicates':
                                risk = edge_data.get('risk_score', 0)
                                contraindicated.append(f"{neighbor} (risk: {risk:.1f})")
                    
                    st.markdown(f"- **{injury.replace('_', ' ').title()}** contraindicates: {', '.join(contraindicated) if contraindicated else 'None in current set'}")
            else:
                st.markdown("✅ No injury constraints - all exercises available")
            
            st.markdown(f"""
            3️⃣ **Multi-Factor Scoring Formula:**
            ```
            For each exercise:
                base_score = (goal_alignment × 0.30) +     
                             (level_match × 0.25) +         
                             (-injury_penalty × 0.30) +     
                             (age_factor × 0.10) +          
                             (variety_bonus × 0.05)
                
                if chat_context_enabled:
                    final_score = base_score × chat_preference_multiplier
            ```
            
            4️⃣ **Graph Edges Traversed:**
            - `{fitness_level}` --[suitable_for]--> Exercises
            - `{fitness_goal}` --[benefits_from]--> Exercises  
            - Injuries --[contraindicates]--> Exercises (negative weights)
            - Exercises --[targets]--> Muscle Groups
            """)
        
        st.markdown("---")
        st.subheader("🎯 Top Recommended Exercises")
        
        # Display top recommendations
        for i, rec in enumerate(recommendations[:6], 1):
            with st.expander(f"#{i} **{rec['name']}** - AI Score: {rec['score']:.3f}", expanded=(i <= 3)):
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown("### Why This Exercise?")
                    
                    if rec['reasons']:
                        for reason in rec['reasons']:
                            if "⚠️" in reason:
                                st.warning(reason)
                            elif "💬" in reason:
                                st.info(reason)
                            else:
                                st.success(f"✓ {reason}")
                    
                    st.markdown("### 🕸️ Knowledge Graph Connections")
                    
                    if fitness_kg.main_graph.has_edge(fitness_goal, rec['name']):
                        effectiveness = fitness_kg.main_graph.get_edge_data(fitness_goal, rec['name']).get('effectiveness', 0)
                        st.info(f"📊 Goal Alignment: {effectiveness*100:.0f}% effective for {fitness_goal.replace('_', ' ')}")
                    
                    st.markdown(f"**💪 Targets:** {', '.join(rec['muscles'])}")
                    
                    injury_risk_display = "🟢 Safe" if rec['injury_risk'] == 1 else "🟡 Moderate" if rec['injury_risk'] == 2 else "🟠 High" if rec['injury_risk'] == 3 else "🔴 Very High"
                    st.markdown(f"**⚠️ Injury Risk:** {injury_risk_display}")
                
                with col2:
                    st.markdown("### 📊 Exercise Details")
                    st.metric("Difficulty", "⭐" * rec['difficulty'])
                    st.metric("Calories/min", f"~{rec['calories']}")
                    st.metric("Type", rec['type'].title())
                    
                    st.markdown("**Score Breakdown:**")
                    score_percentage = (rec['score'] / max([r['score'] for r in recommendations])) * 100
                    st.progress(score_percentage / 100)
        
        # Generate workout plan
        st.markdown("---")
        st.subheader("📋 Personalized Workout Plan")
        
        st.markdown(f"""
        Based on your profile (**{fitness_level}**, goal: **{fitness_goal.replace('_', ' ')}**{' + chat preferences' if use_chat_context else ''}), 
        here's your KG-optimized plan:
        """)
        
        workout_plan = recommendations[:5]
        
        for i, ex in enumerate(workout_plan, 1):
            if fitness_goal == "weight_loss" or ex['type'] == 'cardio':
                sets, reps = 3, "30 seconds" if ex['type'] == 'cardio' else 15
            elif fitness_goal == "muscle_gain":
                sets, reps = 4, 10
            elif fitness_goal == "endurance":
                sets, reps = 3, 20
            else:
                sets, reps = 3, 12
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{i}. {ex['name']}**")
                st.caption(f"Targets: {', '.join(ex['muscles'])}")
            with col2:
                st.markdown(f"**{sets} sets**")
            with col3:
                st.markdown(f"**{reps} reps**")
        
        st.info("💡 **Pro Tip:** Start with 2-3 exercises if you're new, then gradually add more as you progress.")
        
        # Show what was avoided
        if injuries:
            st.markdown("---")
            st.subheader("🚫 Exercises Avoided Due to Your Injuries")
            
            avoided = []
            for injury in injuries:
                if fitness_kg.injury_prevention_graph.has_node(injury):
                    for neighbor in fitness_kg.injury_prevention_graph.neighbors(injury):
                        edge_data = fitness_kg.injury_prevention_graph.get_edge_data(injury, neighbor)
                        if edge_data and edge_data.get('relationship') == 'contraindicates':
                            risk = edge_data.get('risk_score', 0)
                            if risk > 0.5:
                                avoided.append(f"**{neighbor}** (Risk: {risk:.1f}) - due to {injury.replace('_', ' ')}")
            
            if avoided:
                st.warning("The following exercises were excluded or downranked:")
                for av in avoided[:5]:
                    st.markdown(f"- {av}")
            else:
                st.success("✅ No exercises were avoided - your injuries don't restrict the current recommendations!")
    
    else:
        st.info("👆 Click the button above to get personalized recommendations!")
        
        st.markdown("### 🔍 What the Knowledge Graph Will Analyze:")
        
        analysis_points = [
            f"✓ Your **{fitness_level}** level will filter exercises by difficulty",
            f"✓ Your **{fitness_goal.replace('_', ' ')}** goal will prioritize specific exercise types",
            f"✓ Your **age ({age})** will adjust intensity recommendations",
        ]
        
        if injuries:
            analysis_points.append(f"⚠️ Your **{len(injuries)} injury/injuries** will exclude risky exercises")
        else:
            analysis_points.append("✓ No injuries - full exercise library available")
        
        if st.session_state.workout_history:
            analysis_points.append(f"✓ Your **workout history** ({len(st.session_state.workout_history)} sessions) will ensure muscle variety")
        else:
            analysis_points.append("✓ Fresh start - balanced muscle group targeting")
        
        if use_chat_context:
            analysis_points.append("💬 **Chat preferences** will personalize recommendations further")
        
        for point in analysis_points:
            st.markdown(point)

# TAB 3: Progress Analysis
with tab3:
    st.header("📈 Your Progress Analysis (Temporal Graph)")
    
    if st.session_state.workout_history:
        st.subheader("Workout History")
        
        # Show recent workouts
        history_df = pd.DataFrame({
            'Workout #': range(1, len(st.session_state.workout_history) + 1),
            'Exercise': st.session_state.workout_history
        })
        
        st.dataframe(history_df, use_container_width=True)
        
        # Exercise frequency
        st.subheader("Exercise Frequency Analysis")
        exercise_counts = Counter(st.session_state.workout_history)
        
        fig = go.Figure(data=[
            go.Bar(x=list(exercise_counts.keys()), 
                  y=list(exercise_counts.values()),
                  marker_color='lightgreen')
        ])
        fig.update_layout(
            title="Exercises Performed",
            xaxis_title="Exercise",
            yaxis_title="Number of Sessions",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Muscle group coverage
        st.subheader("Muscle Group Coverage")
        
        all_muscles = []
        for ex in st.session_state.workout_history:
            if fitness_kg.main_graph.has_node(ex):
                muscles = fitness_kg.main_graph.nodes[ex].get('muscles', [])
                all_muscles.extend(muscles)
        
        muscle_counts = Counter(all_muscles)
        
        fig = go.Figure(data=[
            go.Pie(labels=list(muscle_counts.keys()), 
                  values=list(muscle_counts.values()),
                  hole=0.3)
        ])
        fig.update_layout(title="Muscle Groups Trained", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on gaps
        st.subheader("🎯 Training Gaps & Recommendations")
        
        all_possible_muscles = ['quads', 'glutes', 'hamstrings', 'chest', 'triceps', 
                               'shoulders', 'core', 'calves', 'full_body']
        trained_muscles = set(all_muscles)
        untrained_muscles = set(all_possible_muscles) - trained_muscles
        
        if untrained_muscles:
            st.warning(f"You haven't trained: {', '.join(untrained_muscles)}")
            
            # Recommend exercises for untrained muscles
            gap_exercises = []
            for muscle in untrained_muscles:
                # Find exercises that target this muscle
                for pred in fitness_kg.main_graph.predecessors(muscle):
                    if fitness_kg.main_graph.nodes[pred].get('node_type') == 'exercise':
                        if fitness_kg.main_graph.get_edge_data(pred, muscle, {}).get('relationship') == 'targets':
                            gap_exercises.append(pred)
            
            if gap_exercises:
                st.info(f"Try these exercises to balance your training: {', '.join(set(gap_exercises[:5]))}")
        else:
            st.success("✅ Great job! You're training all major muscle groups!")
        
        # Temporal graph analysis
        st.subheader("Progress Over Time (Temporal Graph)")
        
        progress_data = fitness_kg.analyze_user_progress(st.session_state.user_id)
        
        if progress_data and len(progress_data) > 1:
            progress_df = pd.DataFrame(progress_data)
            
            fig = go.Figure()
            
            for exercise in progress_df['exercise'].unique():
                ex_data = progress_df[progress_df['exercise'] == exercise]
                fig.add_trace(go.Scatter(
                    x=list(range(len(ex_data))),
                    y=ex_data['reps'],
                    mode='lines+markers',
                    name=exercise,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Rep Count Progress",
                xaxis_title="Workout Session",
                yaxis_title="Reps Completed",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Complete more workouts to see progress trends!")
    
    else:
        st.info("No workout history yet. Start exercising to see your progress!")
        
        # Show demo visualization
        st.subheader("📊 Demo: What You'll See")
        
        demo_data = {
            'Workout': list(range(1, 11)),
            'Squats': [10, 12, 15, 15, 18, 20, 22, 25, 25, 28],
            'Push-ups': [5, 6, 8, 10, 10, 12, 15, 15, 18, 20]
        }
        
        demo_df = pd.DataFrame(demo_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=demo_df['Workout'], y=demo_df['Squats'], 
                                mode='lines+markers', name='Squats'))
        fig.add_trace(go.Scatter(x=demo_df['Workout'], y=demo_df['Push-ups'], 
                                mode='lines+markers', name='Push-ups'))
        
        fig.update_layout(
            title="Example: Progress Over 10 Workouts",
            xaxis_title="Workout Number",
            yaxis_title="Reps",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============= CHATBOT =============
st.markdown("---")
st.subheader("🤖 AI Fitness Coach")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Build context with injury information
injury_info = f"IMPORTANT - User has injuries/limitations: {', '.join(injuries)}. Avoid exercises that stress these areas." if injuries else ""

profile_context = f"""User Profile:
- Age: {age}, Fitness Level: {fitness_level}, Goal: {fitness_goal}
- Current Exercise: {exercise}
{injury_info}
- Recent workouts: {', '.join(st.session_state.workout_history[-3:]) if st.session_state.workout_history else 'None'}

You're a fitness coach. Keep responses SHORT (3-4 sentences) but ALWAYS complete your sentences. Be direct and actionable. If user has injuries, consider them in recommendations."""

# Display existing chat history first
for msg in st.session_state.chat_history:
    role = "You" if msg["role"] == "user" else "AI Coach"
    with st.chat_message(msg["role"]):
        st.markdown(msg['content'])

# Chat input
user_input = st.chat_input("Ask me anything about your workout plan...")

if user_input:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Extract preferences from user input
    user_lower = user_input.lower()
    
    # Detect preferences
    if any(word in user_lower for word in ["like", "love", "prefer", "enjoy", "favorite"]):
        for exercise in ["Squats", "Push-ups", "Lunges", "Plank", "Burpees", "Jumping Jacks"]:
            if exercise.lower() in user_lower and exercise not in st.session_state.chat_preferences['preferred_exercises']:
                st.session_state.chat_preferences['preferred_exercises'].append(exercise)
    
    if any(word in user_lower for word in ["hate", "dislike", "don't like", "avoid", "skip"]):
        for exercise in ["Squats", "Push-ups", "Lunges", "Plank", "Burpees", "Jumping Jacks"]:
            if exercise.lower() in user_lower and exercise not in st.session_state.chat_preferences['disliked_exercises']:
                st.session_state.chat_preferences['disliked_exercises'].append(exercise)
    
    # Detect time constraints
    if any(word in user_lower for word in ["minutes", "min", "time", "quick", "short"]):
        if "10" in user_input or "ten" in user_lower:
            st.session_state.chat_preferences['time_constraints'] = "10 minutes"
        elif "15" in user_input or "fifteen" in user_lower:
            st.session_state.chat_preferences['time_constraints'] = "15 minutes"
        elif "20" in user_input or "twenty" in user_lower:
            st.session_state.chat_preferences['time_constraints'] = "20 minutes"
        elif "30" in user_input or "thirty" in user_lower:
            st.session_state.chat_preferences['time_constraints'] = "30 minutes"
    
    # Detect additional goals
    if "flexibility" in user_lower and "flexibility" not in st.session_state.chat_preferences['additional_goals']:
        st.session_state.chat_preferences['additional_goals'].append("flexibility")
    if "strength" in user_lower and "strength" not in st.session_state.chat_preferences['additional_goals']:
        st.session_state.chat_preferences['additional_goals'].append("strength")
    if "cardio" in user_lower and "cardio" not in st.session_state.chat_preferences['additional_goals']:
        st.session_state.chat_preferences['additional_goals'].append("cardio")
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response - only use last 3 messages for speed
    recent_messages = st.session_state.chat_history[-3:]
    messages = [{"role": "system", "content": profile_context}] + recent_messages
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream the response for faster perceived speed
            response_stream = ollama.chat(
                model="llama2",
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.7,
                    "num_predict": 150,  # Balanced length
                    "top_k": 20,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,  # Avoid repetition
                    "stop": ["User:", "Human:", "\n\nUser", "\n\nHuman"]  # Only stop at conversation breaks
                }
            )
            
            for chunk in response_stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    full_response += chunk['message']['content']
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": full_response
            })
            
        except Exception as e:
            error_msg = f"⚠️ Ollama error. Make sure it's running: `ollama serve`"
            message_placeholder.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg
            })

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
<h4> Advanced Knowledge Graph Fitness System</h4>
<p>Built with NetworkX, Streamlit, MediaPipe & Ollama</p>
</div>
""", unsafe_allow_html=True)
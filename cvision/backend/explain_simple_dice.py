import os
import joblib
import numpy as np
import pandas as pd
from flask import request, jsonify
from pdfminer.high_level import extract_text

# Load model and features
TECH_SKILLS = joblib.load("feature_list.pkl")
SKILL_DESCRIPTIONS = {
    'python': 'Python programming language',
    'machine_learning': 'creating systems that learn from data',
    'deep_learning': 'neural networks and deep architectures',
    'data_analysis': 'insights from structured data',
    'tensorflow': 'building ML models with TensorFlow',
    'pytorch': 'developing models with PyTorch',
    'nlp': 'natural language processing techniques',
    'computer_vision': 'interpreting images and videos with computers',
    'classification': 'labeling data into categories',
    'regression': 'predicting continuous values',
    'sql': 'structured query language for databases',
    'keras': 'neural network modeling using Keras',
    'scikit_learn': 'using scikit-learn for ML tasks',
    # add more if needed...
}

def generate_enhanced_explanations(query, cf_examples, prediction, model):
    current_skills = [skill for skill in TECH_SKILLS if query[skill].iloc[0] > 0.5]
    current_skills_formatted = [skill.replace("_", " ") for skill in current_skills]

    cf_df = cf_examples.cf_examples_list[0].final_cfs_df.copy()
    cf_df.rename(columns={"Match": "Match Percentage"}, inplace=True)
    scenarios = []
    seen_skill_sets = set()

    for idx, cf_row in cf_df.iterrows():
        input_df = pd.DataFrame([cf_row[TECH_SKILLS].values], columns=TECH_SKILLS)
        cf_score = float(np.clip(model.predict(input_df)[0] * 100, 0, 100))
        score_improvement = cf_score - prediction

        added_skills = [
            skill for skill in TECH_SKILLS
            if float(cf_row[skill]) > 0.5 and float(query[skill].iloc[0]) < 0.5
        ]
        added_skills_set = frozenset(added_skills)  # ✅ unique key

        if added_skills_set in seen_skill_sets:
            print(f"⚠️ Duplicate scenario skipped: {sorted(added_skills)}")
            continue
        seen_skill_sets.add(added_skills_set)

        added_skills_info = [{
            "name": skill.replace("_", " "),
            "description": SKILL_DESCRIPTIONS.get(skill, f"knowledge of {skill.replace('_', ' ')}")
        } for skill in added_skills]

        print(f"📊 CF {idx+1}: Score change: {score_improvement:+.2f}% | Added: {[s['name'] for s in added_skills_info]}")

        scenarios.append({
            "added_skills": added_skills_info,
            "new_score": cf_score,
            "improvement": score_improvement
        })


    scenarios.sort(key=lambda x: x["improvement"], reverse=True)

    if scenarios:
        top = scenarios[0]
        summary = (
            f"Your resume currently scores {prediction:.1f}%. "
            f"Best improvement: add {', '.join([s['name'] for s in top['added_skills']])} "
            f"to reach {top['new_score']:.1f}%"
        )
        best_suggestion = f"Potential gain: {'+' if top['improvement'] >= 0 else ''}{top['improvement']:.1f}%"
    else:
        summary = f"Your resume currently scores {prediction:.1f}% match."
        best_suggestion = "No counterfactuals found."

    return {
        "summary": summary,
        "best_suggestion": best_suggestion,
        "current_skills": current_skills_formatted,
        "scenarios": scenarios
    }



"""
Figure 1: Schematic overview of three multi-hospital strategies for EHR
foundation model development.

Generated with gemini-3-pro-image-preview (Nano Banana Pro).
"""

import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = """Create a clean, minimalist scientific schematic diagram showing three strategies for building multi-hospital AI models. Style: flat, geometric, looks like it was made in Microsoft PowerPoint or Adobe Illustrator. NO photorealistic elements, NO gradients, NO flowing curves, NO 3D effects, NO shadows. ALL arrows must be PERFECTLY STRAIGHT lines with simple triangular arrowheads. NO curves, NO bends, NO decorative styles.

ALL arrows across all three panels must have IDENTICAL shape, thickness, and arrowhead style. Only the arrow COLOR varies between panels to code the type of information flowing.

Layout: Three horizontal panels side by side, each labeled (A), (B), (C) at the top.

COMMON ELEMENTS (identical across all three panels):
- 5 small light-gray rectangles at the bottom of each panel, arranged horizontally, labeled "Hospital 1" through "Hospital 5"
- 1 query patient icon at the top of each panel (simple black outline, gender-neutral, labeled "Query Patient")
- Below each panel: a one-line caption in black text

PANEL A - "CENTRALIZED" (left):
- Top: Query Patient icon
- Middle: ONE rectangle labeled "Central Model" with a small database icon (stacked cylinders)
- Bottom: 5 hospital rectangles
- 5 BLUE straight arrows going UP from each hospital to the Central Model, grouped label: "Patient Data" (training phase: data is pooled)
- 1 BLACK straight arrow from Query Patient DOWN to Central Model, labeled "Query"
- 1 BLACK straight arrow from Central Model UP to Query Patient, labeled "Prediction"
- Caption: "Patient data pooled at a central model; one model scores each query"

PANEL B - "FEDERATED" (middle):
- Top: Query Patient icon
- Middle: ONE rectangle labeled "Global Model"
- Bottom: 5 hospital rectangles
- 5 GREEN straight arrows going UP from each hospital to the Global Model, grouped label: "Model Weights" (training phase: weights aggregated, data stays local)
- (NO arrows going back down to hospitals)
- 1 BLACK straight arrow from Query Patient DOWN to Global Model, labeled "Query"
- 1 BLACK straight arrow from Global Model UP to Query Patient, labeled "Prediction"
- Caption: "Model weights aggregated during training; one model scores each query"

PANEL C - "ENSEMBLE" (right):
- Top: Query Patient icon
- Middle: ONE rectangle labeled "Inference Broker" (plays the same structural role as Central Model in A and Global Model in B)
- Bottom: 5 hospital rectangles, each with a small "Local Model" label inside or attached
- NO training-phase arrows (no arrows between hospitals, no arrows to any aggregator during training)
- 1 BLACK straight arrow from Query Patient DOWN to Inference Broker, labeled "Query" (step 1: clinician sends query to broker)
- 5 BLACK straight arrows from Inference Broker DOWN to EACH hospital's Local Model, grouped label: "Dispatched queries" (step 2: broker forwards the query to every model)
- 5 ORANGE straight arrows going UP from each hospital's Local Model BACK to the Inference Broker, grouped label: "Predictions" (step 3: each model returns a prediction)
- 1 BLACK straight arrow from Inference Broker UP to Query Patient, labeled "Averaged Prediction" (step 4: broker averages and returns)
- Caption: "No training-time communication; broker routes each query to all 5 models, predictions averaged"

Color palette (strict, use ONLY these):
- Hospital rectangles and local models: light gray fill (#EEEEEE), black 1pt outline
- Central Model / Global Model / Average box: white fill, dark blue 1.5pt outline
- Query Patient icon: simple black outline person silhouette
- Training-phase arrows Panel A: blue (#1f77b4)
- Training-phase arrows Panel B: green (#2ca02c)
- Training-phase arrows Panel C: NONE (no training communication)
- Query and Prediction arrows Panel A: BLACK
- Query and Prediction arrows Panel B: BLACK
- Query arrows Panel C: BLACK
- Prediction arrows (local models to Average) Panel C: orange (#ff7f0e)
- Final Prediction arrow Panel C: BLACK
- All text: black, clean sans-serif (Arial or Calibri)

ALL arrows are STRAIGHT lines with simple small triangular arrowheads. Same thickness across all panels. No curves, no wavy lines, no decorative ends.

Background: pure white
Aspect ratio: 16:9
Style: professional scientific figure, could be drawn by a human in PowerPoint in 20 minutes. Clean, restrained, symmetric. Think Nature journal illustration but done in PowerPoint."""

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[prompt],
    config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE'],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="2K",
        ),
    ),
)

for part in response.parts:
    if part.text:
        print(part.text)
    elif part.inline_data:
        image = part.as_image()
        image.save("figures/paper_fig1_strategies.jpg")
        print("Saved to figures/paper_fig1_strategies.jpg")

# Save the prompt for reproducibility
with open("figures/paper_fig1_strategies_prompt.txt", "w") as f:
    f.write(f"Model: gemini-3-pro-image-preview\n")
    f.write(f"Aspect ratio: 16:9\n")
    f.write(f"Resolution: 2K\n")
    f.write(f"Date: 2026-04-18\n\n")
    f.write("Prompt:\n")
    f.write(prompt)
print("Prompt saved to figures/paper_fig1_strategies_prompt.txt")

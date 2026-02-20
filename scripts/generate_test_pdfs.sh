#!/bin/bash
# Generate dummy PDFs for testing
# Requires: reportlab (pip install reportlab)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_DIR="$PROJECT_DIR/data/pdfs"

NUM_PDFS=${1:-200}

echo "📄 Generating $NUM_PDFS test PDFs in $DATA_DIR"

mkdir -p "$DATA_DIR"

python3 << EOF
import os
import sys

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
except ImportError:
    print("⚠️ reportlab not installed. Installing...")
    os.system("pip install reportlab")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

import random

DATA_DIR = "$DATA_DIR"
NUM_PDFS = $NUM_PDFS

# Sample content for generating diverse documents
TOPICS = [
    "Machine Learning",
    "Neural Networks", 
    "Natural Language Processing",
    "Computer Vision",
    "Reinforcement Learning",
    "Deep Learning",
    "Data Science",
    "Artificial Intelligence",
    "Big Data Analytics",
    "Cloud Computing",
]

PARAGRAPHS = [
    "This research investigates the application of advanced algorithms to solve complex computational problems. Our methodology involves extensive experimentation with various hyperparameter configurations.",
    "The experimental results demonstrate significant improvements over baseline approaches. Statistical analysis confirms the validity of our findings with p-value < 0.05.",
    "We propose a novel architecture that combines attention mechanisms with convolutional layers. This hybrid approach achieves state-of-the-art performance on benchmark datasets.",
    "The training process utilized gradient descent optimization with adaptive learning rates. Batch normalization and dropout were employed to prevent overfitting.",
    "Future work will explore the scalability of our approach to larger datasets. We also plan to investigate transfer learning techniques for domain adaptation.",
    "The computational requirements include GPU acceleration for efficient matrix operations. Memory optimization techniques were applied to handle large-scale data processing.",
    "Our contributions include a comprehensive analysis of existing methods and a new framework for evaluation. The code and datasets are publicly available for reproducibility.",
    "The model achieves 95.3% accuracy on the test set, outperforming previous methods by 3.2 percentage points. Ablation studies confirm the importance of each component.",
    "Data preprocessing involved tokenization, normalization, and augmentation techniques. Quality assurance measures ensured the integrity of the training data.",
    "The system architecture follows a modular design pattern for flexibility and maintainability. Microservices enable independent scaling of individual components.",
]

print(f"Generating {NUM_PDFS} PDFs...")

for i in range(NUM_PDFS):
    filename = os.path.join(DATA_DIR, f"document_{i:04d}.pdf")
    
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Random topic and content
    topic = random.choice(TOPICS)
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, f"Research Paper on {topic}")
    c.drawString(1 * inch, height - 1.3 * inch, f"Document {i + 1}")
    
    # Content
    c.setFont("Helvetica", 11)
    y_position = height - 2 * inch
    
    # Add multiple paragraphs
    num_paragraphs = random.randint(3, 6)
    selected_paragraphs = random.sample(PARAGRAPHS, min(num_paragraphs, len(PARAGRAPHS)))
    
    for para in selected_paragraphs:
        words = para.split()
        line = ""
        for word in words:
            test_line = line + " " + word if line else word
            if c.stringWidth(test_line, "Helvetica", 11) < width - 2 * inch:
                line = test_line
            else:
                c.drawString(1 * inch, y_position, line)
                y_position -= 14
                line = word
                
                if y_position < 1 * inch:
                    c.showPage()
                    y_position = height - 1 * inch
                    c.setFont("Helvetica", 11)
        
        if line:
            c.drawString(1 * inch, y_position, line)
            y_position -= 20
    
    c.save()
    
    if (i + 1) % 50 == 0:
        print(f"  Generated {i + 1}/{NUM_PDFS} PDFs")

print(f"✅ Generated {NUM_PDFS} PDFs in {DATA_DIR}")
EOF

echo "✅ Done!"

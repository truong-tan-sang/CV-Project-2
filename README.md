# CV Project 2: Gradient Domain Editing & Geometric Transformations

## Structure

```
CV-project/
├── CV_Project_2_Notebook.ipynb    # Main notebook (Google Colab compatible)
├── part1_gradient_domain_editing.py   # Poisson blending
├── part2_geometric_transformations.py # Geometric transforms
├── part3_projective_billboard.py      # Billboard pasting
├── requirements.txt
├── images/                        # Input images (user-provided)
└── results/                       # Output images
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Jupyter Notebook (Recommended)
Open `CV_Project_2_Notebook.ipynb` in Google Colab or Jupyter.

### Option 2: Python Scripts
```bash
python part1_gradient_domain_editing.py
python part2_geometric_transformations.py
python part3_projective_billboard.py
```

## Custom Images

Place your own images in the `images/` folder:
- `images/source.jpg` - Object to blend (Part 1)
- `images/background.jpg` - Target scene (Part 1)
- `images/mask.png` - Binary mask (Part 1, optional)
- `images/input_transform.jpg` - Image for transforms (Part 2)
- `images/scene.jpg` - Scene with planar surface (Part 3)
- `images/content.jpg` - Content to paste (Part 3)

If no custom images are provided, synthetic samples are generated automatically.

# project_structure.py

import os
import json

def document_project_structure():
    """Document the style_classifier_2 project structure and dependencies."""
    
    output = "# Style Classifier 2 Project Documentation\n\n"
    
    # Document Project Structure
    output += "## Project Structure\n\n"
    output += "```\n"
    output += "style_classifier_2/\n"
    output += "├── data/\n"
    output += "│   └── fashion_README.txt\n"
    output += "├── src/\n"
    output += "│   ├── models/\n"
    output += "│   │   ├── darn.py\n"
    output += "│   │   ├── fashionnet.py\n"
    output += "│   │   └── attribute_classifier.py\n"
    output += "│   ├── data_loading.py\n"
    output += "│   ├── train.py\n"
    output += "│   ├── train_fashionnet.py\n"
    output += "│   ├── test_model.py\n"
    output += "│   └── test_fashionnet.py\n"
    output += "├── experiments/\n"
    output += "│   └── fashionnet/\n"
    output += "│       ├── checkpoints/\n"
    output += "│       └── configs/\n"
    output += "```\n\n"

    # Document Dependencies
    output += "## Dependencies\n\n"
    output += "```\n"
    output += "torch\n"
    output += "torchvision\n"
    output += "tqdm\n"
    output += "numpy\n"
    output += "pillow\n"
    output += "tensorboard\n"
    output += "```\n\n"

    # Document Configuration
    output += "## Default Configuration\n\n"
    output += "```json\n"
    output += "{\n"
    output += '    "batch_size": 32,\n'
    output += '    "num_epochs": 20,\n'
    output += '    "learning_rate": 0.0001,\n'
    output += '    "weight_decay": 0.0001,\n'
    output += '    "subset_size": 0.1,\n'
    output += '    "loss_weights": {\n'
    output += '        "category_weight": 1.0,\n'
    output += '        "attribute_weight": 1.0,\n'
    output += '        "landmark_weight": 0.01\n'
    output += '    }\n'
    output += "}\n"
    output += "```\n"

    # Write to file
    with open('project_structure.md', 'w') as f:
        f.write(output)

if __name__ == "__main__":
    document_project_structure()
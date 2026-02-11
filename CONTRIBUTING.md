# Contributing to RL-Based TSR Frequency Domain Attacker

First off, thank you for considering contributing to this project! 🎉

The following is a set of guidelines for contributing to the RL-Based Traffic Sign Recognition project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [How Can I Contribute?](#how-can-i-contribute)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by mutual respect and professionalism. By participating, you are expected to uphold this standard.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

### Prerequisites

Before you begin, ensure you have:

1. **Git** and **Git LFS** installed
2. **Python 3.8+** installed
3. A **GitHub account**
4. Familiarity with **PyTorch** and deep learning concepts

### Setup Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR-USERNAME/RL-Based-TSR-Frequency-Domain-Attacker.git
cd RL-Based-TSR-Frequency-Domain-Attacker
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/ORIGINAL-OWNER/RL-Based-TSR-Frequency-Domain-Attacker.git
```

4. Create a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Verify everything works:

```bash
python test_workflow.py
```

## Development Workflow

### Branching Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Urgent production fixes

### Working on a Feature

1. **Update your local main branch:**

```bash
git checkout main
git pull upstream main
```

2. **Create a feature branch:**

```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes** and commit regularly

4. **Keep your branch updated:**

```bash
git fetch upstream
git rebase upstream/main
```

5. **Push your branch:**

```bash
git push origin feature/your-feature-name
```

6. **Open a Pull Request** on GitHub

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When submitting a bug report, include:**

- **Clear title and description**
- **Exact steps to reproduce** the problem
- **Expected vs actual behavior**
- **Screenshots** if applicable
- **Environment details:**
  - OS and version
  - Python version
  - PyTorch version
  - CUDA version (if using GPU)

**Template:**

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Windows 11]
- Python: [e.g., 3.11.4]
- PyTorch: [e.g., 2.0.1]
- CUDA: [e.g., 11.8]

**Additional context**
Any other context about the problem.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues.

**When suggesting an enhancement, include:**

- **Clear title and description**
- **Use case** - why is this enhancement useful?
- **Proposed solution** if you have one
- **Alternative solutions** you've considered
- **Impact** on existing functionality

### Adding New Features

We welcome new features! Here are some ideas:

#### 1. New Adversarial Attack Types

Add new attack methods in `TSR_Models/generate_adv_dataset.py`:

```python
def apply_adversarial_patch(image, patch_type='occlusion', size=20):
    # ... existing code ...
    
    elif patch_type == 'your_new_attack':
        # Your implementation here
        # Example: Fog effect
        fog = np.random.normal(0.8, 0.1, image.shape)
        image = image * 0.7 + fog * 0.3
        image = np.clip(image, 0, 1)
    
    return torch.tensor(image.transpose((2, 0, 1)))
```

**Checklist:**
- [ ] Implement the attack function
- [ ] Add docstring explaining the attack
- [ ] Test on sample images
- [ ] Add to attack types list
- [ ] Update README with examples
- [ ] Include visual examples in PR

#### 2. New Model Architectures

Create a new model file in `TSR_Models/`:

```python
# TSR_Models/your_model.py
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, num_classes=47):
        super(YourModel, self).__init__()
        # Your architecture here
    
    def forward(self, x):
        # Forward pass
        return x
```

Add to `models_factory.py`:

```python
def load_your_model(num_classes, device):
    from your_model import YourModel
    model = YourModel(num_classes=num_classes)
    # Load weights if available
    return model.to(device)
```

**Checklist:**
- [ ] Implement model architecture
- [ ] Add comprehensive docstrings
- [ ] Create loader function
- [ ] Add model tests
- [ ] Benchmark performance
- [ ] Update documentation
- [ ] Include architecture diagram/description

#### 3. Evaluation Metrics

Add new metrics in `TSR_Models/benchmark_and_eval.py`:

```python
from sklearn.metrics import your_metric

def evaluate_model_comprehensive(model, dataloader, device):
    # ... existing metrics ...
    
    # Add your metric
    your_score = your_metric(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'your_metric': your_score,
        # ... other metrics
    }
```

#### 4. Data Augmentation Techniques

Enhance the dataset pipeline with new augmentations.

#### 5. Training Improvements

- Learning rate schedulers
- Advanced optimizers
- Regularization techniques
- Mixed precision training

### Code Review

All submissions require review. We use GitHub pull requests for this purpose.

**As a Reviewer:**
- Be respectful and constructive
- Focus on the code, not the person
- Explain your reasoning
- Suggest alternatives when possible

**As a Contributor:**
- Be open to feedback
- Respond to comments promptly
- Make requested changes or explain why not

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length:** 100 characters (not 79)
- **Indentation:** 4 spaces (no tabs)
- **Quotes:** Single quotes for strings, double for docstrings
- **Imports:** Grouped and sorted (stdlib, third-party, local)

```python
# Good
import os
import sys

import torch
import numpy as np

from lisa import LISA
from models_factory import load_cnn_3spt


def my_function(param1, param2):
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    return result
```

### Docstring Format

Use Google-style docstrings:

```python
def apply_adversarial_patch(image, patch_type='occlusion', size=20):
    """
    Applies an adversarial modification to the image.
    
    This function implements various adversarial attack types including
    occlusion, light reflection, and noise perturbations.
    
    Args:
        image (torch.Tensor): Input image tensor with shape (C, H, W).
        patch_type (str): Type of attack. Options: 'occlusion', 'light',
            'perturbation', 'localized_perturbation'. Default: 'occlusion'.
        size (int): Size of the adversarial patch in pixels. Automatically
            capped at 25% of image dimensions. Default: 20.
    
    Returns:
        torch.Tensor: Modified image tensor with same shape as input.
    
    Raises:
        ValueError: If patch_type is not recognized.
    
    Example:
        >>> image = torch.rand(3, 32, 32)
        >>> adv_image = apply_adversarial_patch(image, 'occlusion', 15)
        >>> adv_image.shape
        torch.Size([3, 32, 32])
    """
    pass
```

### Code Organization

- Keep functions focused and single-purpose
- Use meaningful variable names
- Avoid magic numbers - use named constants
- Comment complex logic
- Wrap test code in `if __name__ == "__main__"`

```python
# Good
MAX_PATCH_SIZE_RATIO = 0.25
MIN_IMAGE_DIMENSION = 32

def calculate_patch_size(image_size, requested_size):
    """Calculate safe patch size based on image dimensions."""
    max_allowed = int(MAX_PATCH_SIZE_RATIO * min(image_size))
    return min(requested_size, max_allowed)


if __name__ == "__main__":
    # Test code here
    pass
```

### Testing

**All new features should include tests:**

```python
# test_your_feature.py
import unittest
import torch
from your_module import your_function

class TestYourFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = torch.rand(3, 32, 32)
    
    def test_basic_functionality(self):
        """Test basic use case."""
        result = your_function(self.test_image)
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with very small image
        small_image = torch.rand(3, 8, 8)
        result = your_function(small_image)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
```

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(attacks): Add fog adversarial attack type

Implemented fog weather simulation as a new adversarial attack.
The attack reduces visibility by blending the image with gaussian noise.

- Added fog_attack() function
- Updated generate_adv_dataset.py
- Added tests and examples
- Updated documentation

Closes #42
```

```
fix(model): Correct SAG-ViT input normalization

Fixed incorrect mean/std values for ImageNet normalization
in SAG-ViT model preprocessing.

Before: mean=[0.5, 0.5, 0.5]
After: mean=[0.485, 0.456, 0.406]
```

```
docs(readme): Update installation instructions for Windows

Added PowerShell-specific commands and troubleshooting for
Windows users experiencing PATH issues with Git.
```

## Pull Request Process

### Before Submitting

1. **Test your changes:**
   ```bash
   python test_workflow.py
   python test_models.py
   ```

2. **Verify code style:**
   ```bash
   # Install flake8 if needed
   pip install flake8
   flake8 your_modified_files.py
   ```

3. **Update documentation:**
   - Update README.md if adding features
   - Add docstrings to new functions
   - Update CHANGELOG.md (if exists)

4. **Check for conflicts:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tested locally with `test_workflow.py`
- [ ] Tested locally with `test_models.py`
- [ ] Added new tests for new features
- [ ] All existing tests pass

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Screenshots (if applicable)
Add screenshots for UI changes or visual results

## Additional Notes
Any additional information reviewers should know
```

### PR Review Process

1. **Automated checks** will run (if configured)
2. **At least one maintainer** must review
3. **Address feedback** from reviewers
4. **Squash commits** if requested
5. **Merge** once approved

### After Merge

1. Delete your feature branch
2. Pull the updated main branch
3. Celebrate! 🎉

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## Questions?

- **Open an issue** for general questions
- **Check existing issues** first
- **Be specific** in your questions
- **Provide context** about what you're trying to achieve

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Git Flow Guide](https://guides.github.com/introduction/flow/)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)
- [Code Review Best Practices](https://google.github.io/eng-practices/review/)

---

Thank you for contributing! Your efforts help make this project better for everyone. 🚀

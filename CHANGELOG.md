# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-11

### Added
- Initial public release of RL-Based TSR Frequency Domain Attacker
- Four pre-trained model architectures:
  - CNN-3SPT with Spatial Transformer Network
  - SAG-ViT (Spatial-Aware Graph Vision Transformer)
  - ResNet50 with transfer learning
  - ViT-B/16 Vision Transformer
- Adversarial attack generation framework with four attack types:
  - Occlusion attacks
  - Light reflection simulation
  - Gaussian noise perturbations
  - Localized perturbations
- Automated dataset generation pipeline
- LISA traffic sign dataset integration with automatic download
- Model benchmarking and evaluation tools
- Transfer learning pipeline
- Comprehensive test suite (`test_workflow.py`, `test_models.py`)
- Pre-trained model weights via Git LFS (636 MB total):
  - cnn_3spt.pth (22.4 MB)
  - sag_vit.pth (25.9 MB)
  - vit_pretrained_best.pth (327.5 MB)
  - resnet_cnn_model.pth (42.8 MB)
- Complete documentation:
  - README.md with quick start guide
  - CONTRIBUTING.md with contribution guidelines
  - README_WORKFLOW.md with detailed workflow documentation
  - SETUP_INSTRUCTIONS.md with installation guide
- MIT License

### Infrastructure
- Git LFS integration for large model files
- Python virtual environment support
- Automated dependency management with requirements.txt
- .gitignore configured for Python ML projects

### Documentation
- Comprehensive README with badges, examples, and usage
- Contributing guidelines with code style and PR process
- Detailed API documentation in code
- Troubleshooting guide
- Performance benchmarks

### Testing
- Workflow verification script
- Model loading and inference tests
- Dataset generation validation

## [Unreleased]

### Planned Features
- Additional adversarial attack types (fog, rain, snow)
- Real-time inference demo
- Model ensemble methods
- Quantization and optimization for edge deployment
- Extended dataset support (GTSRB, TT100K)
- Advanced training techniques (knowledge distillation)
- Web interface for model testing
- Docker containerization
- CI/CD pipeline integration

## Version History

### Version Numbering
- **Major version**: Incompatible API changes
- **Minor version**: New features (backward compatible)
- **Patch version**: Bug fixes (backward compatible)

---

For detailed changes in each release, see the [Releases](https://github.com/yourusername/RL-Based-TSR-Frequency-Domain-Attacker/releases) page.

![Counterfit](counterfit.png "Counterfit")

## About
Counterfit is a generic automation layer for assessing the security of machine learning systems. It brings several existing adversarial frameworks under one tool, or allows users to create their own. 

### Requirements
- Ubuntu 18.04+
- Python 3.7 or 3.8
- Windows is supported by Counterfit, but not necessarily officially supported by each individual framework. 
- On Windows the [Visual C++ 2019 redistributable](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) is required

## Quick Start
1. Create and activate a virtual environment, `python -m venv cf-venv`
2. Install Counterfit: `git clone --depth 1 -b develop/1.1 https://github.com/Azure/counterfit.git`
3. `cd counterfit`
4. `pip install .[dev]` (for all frameworks)
5. Go checkout the examples folder. 

Notes: 
- Windows requires C++ build tools
- If textattack has been installed, it will initialize by downloading nltk data
## Acknowledgments
Counterfit leverages excellent open source projects, including,

- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [TextAttack](https://github.com/QData/TextAttack)
- [Augly](https://github.com/facebookresearch/AugLy)


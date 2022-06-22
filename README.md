![Counterfit](counterfit.png "Counterfit")




## About
Counterfit is a red team tool for finding and exploiting machine learning systems. This is a rewrite of the Counterfit tool found here. 

### Requirements
- Ubuntu 18.04+
- Python 3.7+
- Windows is supported by Counterfit, but not necessarily officially supported by each individual framework. Choose your own adventure
- On Windows the [Visual C++ 2019 redistributable](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) is required

## Quick Start
It's highly recommended you use a virtual env of some kind.
```
$> python -m venv cf-venv
$> git clone https://github.com/moohax/counterfit.git
$> cd counterfit
```

### Linux
```
$> source /cf-venv/bin/activate
```

### Windows
```
$> cf-venv\Scripts\activate.[bat, ps1]
```

### Install 
```
$> pip install .
```

### Notes
Most ML frameworks will initialize a `.` folder somewhere. These folders are used to cache data and models. For example, `textattack` will pull NLTK packages into a local directory. 


## Acknowledgments

- Logo by [@monoxgas](https://twitter.com/monoxgas)

Counterfit leverages excellent open source projects, including,

- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [TextAttack](https://github.com/QData/TextAttack)
- [Augly](https://github.com/facebookresearch/AugLy)



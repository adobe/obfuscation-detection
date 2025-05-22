# Command Obfuscation Detection

This project currently only supports cmd.exe command obfuscation detection on Windows. In a previous iteration of this project, we used deep learning. Now, we have shifted the approach towards XGBoost instead.

- Blog post: https://medium.com/adobetech/using-deep-learning-to-better-detect-command-obfuscation-965b448973e0
- Pip package: https://pypi.org/project/obfuscation-detection/

## Quick Installation

You can install our package through pip!
```
pip install obfuscation-detection
```

This is a basic usage of our package:
```
from obfuscation_detection import ObfuscationDetectionClassifier

model = ObfuscationDetectionClassifier()
commands = ['cmd.exe /c "echo Invoke-DOSfuscation"',
            'cm%windir:~ -4, -3%.e^Xe,;^,/^C",;,S^Et ^^o^=fus^cat^ion&,;,^se^T ^ ^ ^B^=o^ke-D^OS&&,;,s^Et^^ d^=ec^ho I^nv&&,;,C^Al^l,;,^%^D%^%B%^%o^%"',
            'cat /etc/passwd']
y = model.predict(commands)
y_prob = model.predict_proba(commands)

# 1 is obfuscated, 0 is non-obfuscated
print(y) # [0, 1, 0]
print(y_prob)
```

## Usage
1. Install python dependencies: `pip install -r requirements.txt`

2. For quick usage, give a .csv file with column `commands` and you can run the commands through the model: `python obfuscation_detection/main.py --filename commands.csv`

3. You can also write your own scripts to use the model class directly: `python obfuscation_detection/model.py`

### Contributing

Contributions are welcomed! Read the [Contributing Guide](./CONTRIBUTING.md) for more information.

### Licensing

This project is licensed under the Apache V2 License. See [LICENSE](LICENSE) for more information.

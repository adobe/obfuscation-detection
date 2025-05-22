import numpy as np
import os
from importlib.resources import files
from xgboost import XGBClassifier

class ObfuscationDetectionClassifier:
    MAX_INPUT_LENGTH = 8192

    MODEL_PARAMS = {
        "booster": "gbtree",
        "device": "cpu",
        "random_state": 42,
        "n_estimators": 50,
        "learning_rate": 0.3,
        "gamma": 0,
        "max_depth": 6,
        "min_child_weight": 1,
        "max_delta_step": 0,
        "subsample": 1,
        "sampling_method": "uniform",
        "colsample_bylevel": 1,
        "colsample_bynode": 1,
        "colsample_bytree": 1,
        "reg_lambda": 1,
        "reg_alpha": 0,
        "tree_method": "auto",
        "scale_pos_weight": 1,
        "max_leaves": 0,
        "max_bin": 256,
        "num_parallel_tree": 1,
        "objective": "binary:logistic",
        "validate_parameters": True,
    }

    CHAR_DICT = {
        ' ': 9,
        '"': 22,
        '#': 93,
        '$': 69,
        '%': 82,
        '&': 77,
        "'": 70,
        '(': 74,
        ')': 73,
        '*': 91,
        '+': 85,
        ',': 68,
        '-': 27,
        '.': 34,
        '/': 45,
        '0': 28,
        '1': 44,
        '2': 47,
        '3': 53,
        '4': 37,
        '5': 60,
        '6': 62,
        '7': 66,
        '8': 46,
        '9': 61,
        ':': 49,
        ';': 72,
        '<': 81,
        '=': 71,
        '>': 78,
        '?': 92,
        '@': 90,
        'A': 0,
        'B': 2,
        'C': 4,
        'D': 30,
        'E': 31,
        'F': 38,
        'G': 16,
        'H': 23,
        'I': 8,
        'J': 63,
        'K': 67,
        'L': 50,
        'M': 36,
        'N': 43,
        'O': 55,
        'P': 39,
        'Q': 14,
        'R': 56,
        'S': 32,
        'T': 42,
        'U': 25,
        'V': 76,
        'W': 59,
        'X': 75,
        'Y': 57,
        'Z': 35,
        '[': 88,
        '\\': 5,
        ']': 87,
        '^': 64,
        '_': 41,
        '`': 89,
        'a': 11,
        'b': 24,
        'c': 12,
        'd': 19,
        'e': 1,
        'f': 51,
        'g': 3,
        'h': 52,
        'i': 17,
        'j': 58,
        'k': 33,
        'l': 15,
        'm': 26,
        'n': 18,
        'o': 6,
        'p': 21,
        'q': 84,
        'r': 13,
        's': 10,
        't': 7,
        'u': 29,
        'v': 48,
        'w': 20,
        'x': 54,
        'y': 40,
        'z': 65,
        '{': 79,
        '|': 86,
        '}': 80,
        '~': 83
    }

    def __init__(self):
        self.model = XGBClassifier(**ObfuscationDetectionClassifier.MODEL_PARAMS)
        try:
            # Works when installed as a package (via pip)
            model_file = str(files("obfuscation_detection.models") / "od-cmdexe-xgb-model.json")
        except (ModuleNotFoundError, AttributeError):
            # Fallback for local run: find relative path manually
            model_file = os.path.join(os.path.dirname(__file__), "models", "od-cmdexe-xgb-model.json")
        self.model.load_model(model_file)
        
    def vectorize_commands(self, commands):
        VECTOR_LENGTH = len(ObfuscationDetectionClassifier.CHAR_DICT) + 3
        X = []
        for command in commands:
            command_vectorized = [0.0 for _ in range(VECTOR_LENGTH)]
            command_vectorized[VECTOR_LENGTH - 1] = float(len(command)) # command line length
            # shorten command
            command = command[:ObfuscationDetectionClassifier.MAX_INPUT_LENGTH]
            command_chars = {s: command.count(s) for s in set(command)}
            unk_chars = 0
            alnum_chars = 0

            for char in command_chars:
                if char in ObfuscationDetectionClassifier.CHAR_DICT:
                    command_vectorized[ObfuscationDetectionClassifier.CHAR_DICT[char]] = command_chars[char] / len(command) # char frequency
                elif not (char.isalnum() or char in ('!', '\n', '\r', '\t')):
                    unk_chars += command_chars[char]

                if char.isalnum():
                    alnum_chars += command_chars[char]
                # else continue bc one of character exclusions
            
            command_vectorized[VECTOR_LENGTH - 2] = unk_chars / len(command) # unknown chars
            command_vectorized[VECTOR_LENGTH - 3] = alnum_chars / len(command) # alphanumeric chars
            X.append(command_vectorized)
        return np.array(X)

    def predict(self, commands):
        X = self.vectorize_commands(commands)
        return self.model.predict(X)
    
    def predict_proba(self, commands):
        X = self.vectorize_commands(commands)
        return self.model.predict_proba(X)

if __name__ == "__main__":
    model = ObfuscationDetectionClassifier()
    commands = [
        'cmd.exe /c "echo Invoke-DOSfuscation"',
        'cm%windir:~ -4, -3%.e^Xe,;^,/^C",;,S^Et ^^o^=fus^cat^ion&,;,^se^T ^ ^ ^B^=o^ke-D^OS&&,;,s^Et^^ d^=ec^ho I^nv&&,;,C^Al^l,;,^%^D%^%B%^%o^%"',
        'cat /etc/passwd'
    ]
    y = model.predict(commands)
    y_prob = model.predict_proba(commands)
    
    print(y)
    print(y_prob)

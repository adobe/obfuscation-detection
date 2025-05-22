import argparse
import pandas as pd
from model import ObfuscationDetectionClassifier

def main():
    parser = argparse.ArgumentParser(description="Provide a .csv file with commands")
    parser.add_argument("--filename", help="Path to the input file")

    args = parser.parse_args()
    filename = args.filename

    commands = pd.read_csv(filename, sep='\t')['commands'].tolist()
    model = ObfuscationDetectionClassifier()
    y = model.predict(commands)

    print(y)

if __name__ == "__main__":
    main()
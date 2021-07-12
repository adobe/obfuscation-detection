# Obfuscation Detection

## How to test on a real script
1. Grab the best model checkpoint from [Sharepoint](https://adobe-my.sharepoint.com/personal/vikrakum_adobe_com1/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fvikrakum%5Fadobe%5Fcom1%2FDocuments%2Ffile%2Dsharing) and place the file in a new directory `./models/` in this repo.

2. Install the dependencies needed. I used Anaconda with Python 3.8.10.

3. Run `python --model cnn --model_file best-cnn1-128-fc-1024-1024-dropout-80.pth --run <SCRIPT_PATH>`
- As of now, it will only run one script at a time.

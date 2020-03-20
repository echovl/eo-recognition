import os

os.system("docker build -t eo-recognition .")

os.system("docker run -p 8000:8080 eo-recognition serve")

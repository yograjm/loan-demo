# python installed
FROM python:3.10-slim

# requirements installed
# ADD dataset/ dataset/
ADD trained_model/*.pkl trained_model/
ADD requirements/ requirements/

RUN pip install -r requirements/api_requirements.txt

# all the related files .py
ADD *.py ./

# expose the port where appl.is running
EXPOSE 7860
EXPOSE 8080

# command to start the appl.
CMD ["python", "app2.py"]

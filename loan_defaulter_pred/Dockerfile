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
CMD ["python", "app.py2"]


# docker build -t loan_default_pred .

# docker tag loan_default_pred yrajm1997/loan_default_pred

# docker run -d -it -p 7860:7860 --name=myapp loan_default_pred


# Login step

# docker login -u yrajm1997
# docker login (location of your registry, credentials, etc)


# To pull and us the image
# docker login (location of your registry, credentials, etc)
# docker pull ..

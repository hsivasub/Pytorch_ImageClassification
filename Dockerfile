FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install opencv-python-headless

RUN pip install -r requirements.txt

COPY  new_model/* ${LAMBDA_TASK_ROOT}

CMD ["inferencing_with_lambda.lambda_handler"]

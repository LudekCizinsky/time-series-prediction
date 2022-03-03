## About 
This small project is an assignment in the course `Large Scale Data Analysis`
taken at [IT University of Copenhagen](https://www.itu.dk/). Its goal is to
predict power production based on the speed and direction of the wind.

## Prerequisites

### Python3
First, you need to have `python3` installed. If not, you can do so [here](https://www.python.org/downloads/). 

### Virtual env (optional but recommended)

First, navigate to the folder where you are storing your venvs and then use venv to create 
virtual env for this project as follows: 
    
```
python3 -m venv [name of venv]
```

Now, you can activate the venv through the command: 

```
source [name of env]/bin/activate
```

Deactivate through:

```
deactivate
```

Lastly, make sure your pip is updated: `pip install --upgrade pip`. If you do not have pip installed, try to google how to install it based on your `os`.

### Clone repo

Clone the repository to your desired folder locally.

### Install the needed packages
Run the following command from the root of the cloned directory to install all dependencies 
into the virtual venv:

```
pip install -r requirements.txt
```

## Quickstart

### Allow execution of given bash script

Next, you need to allow execution of the bash script which allows you to run the
project from your command line assuming you have `bash` installed. Run the
following command from the root of your repo:

```
chmod +x run.sh
```

### Run the pipeline
Finally, you can run the whole pipeline by executing:

```
./run.sh
``` 


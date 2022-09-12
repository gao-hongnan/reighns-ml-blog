## Gearing Up For Development

### Set Up Main Directory (IDE)

Let us assume that we are residing in our root folder `~/gaohn` and
we want to create a new project called **YOLOX**, we can do as follows:

```bash title="creating main directory" linenums="1"
~/gaohn       $ mkdir YOLOX
~/gaohn       $ cd YOLOX
~/gaohn/YOLOX $ code .        # (1)
```
            
1.  Open the project directory in Visual Studio Code. To change appropriately if using different IDE.

If you are cloning a repository to your local folder **YOLOX**, you can further do:

```bash title="cloning repository" linenums="1"
~/gaohn/YOLOX $ git clone https://github.com/Megvii-BaseDetection/YOLOX.git .
```

where `.` means cloning to the current directory.

### Set Up Virtual Environment

Follow the steps below to set up a virtual environment for your development.

=== "Windows"

    ```bash title="venv" linenums="1"
    ~/gaohn/YOLOX        $ python -m venv <name of virtual env>                      # (1)
    ~/gaohn/YOLOX        $ .\venv\Scripts\activate                                   # (2)
    ~/gaohn/YOLOX (venv) $ python -m pip install --upgrade pip setuptools wheel      # (3)
    ```

    1.  Create virtual environment named `venv` in the current directory.
    2.  Activate virtual environment.
    3.  Upgrade pip, setuptools and wheel.

=== "macOS M1"

    ```bash title="venv" linenums="1"
    ~/gaohn/YOLOX        $ pip3 install virtualenv 
    ~/gaohn/YOLOX        $ virtualenv <name of virtual env>                                          
    ~/gaohn/YOLOX        $ source .\venv\bin\activate
    ~/gaohn/YOLOX (venv) $ python3 -m pip install --upgrade pip setuptools wheel
    ```

=== "Linux"

    ```bash title="venv" linenums="1"
    ~/gaohn/YOLOX        $ sudo apt install python3.8 python3.8-venv python3-venv 
    ~/gaohn/YOLOX        $ python3 -m venv <name of virtual env>                                 
    ~/gaohn/YOLOX        $ source .\venv\bin\activate                                 
    ~/gaohn/YOLOX (venv) $ python3 -m pip install --upgrade pip setuptools wheel      
    ```

You should see the following directory structure:

```tree title="main directory tree" linenums="1"
YOLOX/
└── venv/
```

### Requirements and Setup

!!! note
    For small projects, we can have `requirements.txt` and just run `pip install -r requirements.txt`.
    For larger projects, we can add a `setup.py` file.

    In short, `requirements.txt` specifies the dependencies of your project, and [`setup.py`](https://
    stackoverflow.com/questions/60145069/what-is-the-purpose-of-setup-py) file informs you about the 
    module or package-dependencies you are about to install has been packaged and distributed with
    Distutils, which is the standard for distributing Python Modules. You can skip `setup.py` if you
    are just using `requirements.txt` to install dependencies.

    Refer to [madewithml](https://madewithml.com/courses/mlops/packaging/)'s
    **requirements** and **setup** section for more details.      

```bash title="creating requirements" linenums="1"
~/gaohn/YOLOX (venv) $ touch requirements.txt setup.py                                  
~/gaohn/YOLOX (venv) $ pip install -e .                           
```

Something worth taking note is when you download PyTorch Library, there is a dependency link since we are downloading cuda directly, you may execute as such:

```bash
~/gaohn/YOLOX (venv) $ pip install -e . -f https://download.pytorch.org/whl/torch_stable.html
```

If you further specify packages for **development**, **testing** and **documentation** in `setup.py`,
you can choose which version to install:

```bash title="installing packages" linenums="1"
~/gaohn/YOLOX (venv) $ python -m pip install -e ".[dev]"                       # installs required + dev packages
~/gaohn/YOLOX (venv) $ python -m pip install -e ".[test]"                      # installs required + test packages
~/gaohn/YOLOX (venv) $ python -m pip install -e ".[docs_packages]"             # installs required documentation packages
```

!!! Danger "TODO"
    To add in new examples here.

      
You should see the following directory structure:

```tree title="main directory tree" linenums="1"
YOLOX/
├── venv/
├── requirements.txt
└── setup.py
```
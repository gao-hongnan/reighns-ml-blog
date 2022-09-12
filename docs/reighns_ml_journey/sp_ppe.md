## Gearing Up For Development

### Set Up Main Directory (IDE)

Let us assume that we are residing in our root folder `~/gaohn` and
we want to create a new project called **YOLOX**, we can do as follows:

```bash title="creating main directory" linenums="1"
~/gaohn       $ mkdir YOLOX
~/gaohn       $ cd YOLOX
~/gaohn/YOLOX $ code .                      # (1)
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
~/gaohn/YOLOX  (venv) $ touch requirements.txt setup.py                                  
~/gaohn/YOLOX  (venv) $ pip install -e .                           
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

For this version of YOLOX, use the commands below to do installation:

=== "Windows"

    ```bash title="install" linenums="1"
    git clone git@github.com:Megvii-BaseDetection/YOLOX.git
    cd YOLOX
    pip install -U pip && pip install -r requirements.txt
    pip install -v -e .  # or  python setup.py develop
    pip install cython
    # can install pycocotools==2.0.2 manually
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    pip install pycocotools==2.0.2
    # not sure why my com not playing nice with cuda 11.6 but only work with this for now
    # even cu102 will not work
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip install setuptools==59.5.0 # for https://stackoverflow.com/questions/70520120/attributeerror-module-setuptools-distutils-has-no-attribute-version
    pip install tensorboard
    pip install wandb
    ```

=== "macOS M1"

    ```bash title="install" linenums="1"
    ...
    ```

## Get Raw Data

The incoming raw data is a zip file named `sp_ppe_all_combination_images.zip` and contains 1,600 images
alongside a csv file named `all_annotations.csv` which contains the bounding box coordinates and
its corresponding class labels.

Our goal is to download these data to our local machine (main directory).

```tree title="main directory tree" linenums="1"
YOLOX/
├── venv/
├── datasets/
    └── sp_ppe_data/
        ├── sp_ppe_all_images/
        └── sp_ppe_all_annotations/
|── ...
├── requirements.txt
└── setup.py
```


where `sp_ppe_all_images` contains 1,600 images and `sp_ppe_all_annotations` contains 1,600
corresponding annotations in `.xml` format. We will touch on that later.

1. First, we create a folder named `datasets` in the root directory of the project alongside
    its sub-folders `sp_ppe_data` and `sp_ppe_all_images` and `sp_ppe_all_annotations`.

    ```bash title="creating datasets folder" linenums="1"
    ~/gaohn/YOLOX (venv) $ mkdir -p datasets/sp_ppe_data 
    ~/gaohn/YOLOX (venv) $ mkdir -p datasets/sp_ppe_data/sp_ppe_all_images
    ~/gaohn/YOLOX (venv) $ mkdir -p datasets/sp_ppe_data/sp_ppe_all_annotations 
    ```

    Take note that in this YOLOX repo, the `datasets` folder is already there when
    you clone the repo, but having the command `-p` allows you to create the subfolder
    without raising an error.

2. To download the data into `datasets/sp_ppe_data`, we can issue the below commands:
    
    === "Windows"

        ```bash title="download raw data" linenums="1"
        ~/gaohn/YOLOX (venv) $ wget -P <destination folder> <url> 
        ~/gaohn/YOLOX (venv) $ tar xf <zip file> -C <destination folder>
        ```

    === "macOS M1"

        ```bash title="download raw data" linenums="1"
        ~/gaohn/YOLOX (venv) $ wget -P <destination folder> <url> 
        ~/gaohn/YOLOX (venv) $ unzip <zip file> -d <destination folder>
        ```

    where the `<url>` for the raw data is 
    `https://storage.googleapis.com/peekingduck/data/sp_ppe_all_combination_images.zip`.
    
    More concretely, we have:

    ```bash title="download raw data windows" linenums="1"
    ~/gaohn/YOLOX (venv) $ wget -P datasets/sp_ppe_data/sp_ppe_all_images https://storage.googleapis.com/peekingduck/data/sp_ppe_all_combination_images.zip
    ~/gaohn/YOLOX (venv) $ tar xf datasets/sp_ppe_data/sp_ppe_all_images/sp_ppe_all_combination_images.zip -C datasets/sp_ppe_data/sp_ppe_all_images
    ~/gaohn/YOLOX (venv) $ rm datasets/sp_ppe_data/sp_ppe_all_images/sp_ppe_all_combination_images.zip
    ```

3. Now we want to move our `all_annotations.csv` file into `datasets/sp_ppe_data/sp_ppe_all_annotations`.
    We can do this by issuing the following command:


    === "Windows"

        ```bash title="move files" linenums="1"
        ~/gaohn/YOLOX (venv) $ move <source file> <destination folder>
        ```

    === "macOS"

        ```bash title="move files" linenums="1"
        ~/gaohn/YOLOX (venv) $ mv <source file> <destination folder>
        ```

    === "Linux"

        ```bash title="move files filesnv" linenums="1"
        ~/gaohn/YOLOX (venv) $ mv <source file> <destination folder> 
        ```

    So in our case, it is simply:

    ```bash title="move files windows" linenums="1"
    ~/gaohn/YOLOX (venv) $ move datasets/sp_ppe_data/sp_ppe_all_images/all_annotations.csv datasets/sp_ppe_data/sp_ppe_all_annotations
    ```

4. The above steps can be a good opportunity to introduce a very basic shell script.

    ```bash
    #!/bin/bash
    # YOLOv5 🚀 by Ultralytics, GPL-3.0 license
    # Download COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017)
    # Example usage: bash data/scripts/get_coco128.sh
    # parent
    # ├── yolov5
    # └── datasets
    #     └── coco128  ← downloads here

    # Download/unzip images and labels
    d='../datasets' # unzip directory
    url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
    f='coco128.zip' # or 'coco128-segments.zip', 68 MB
    echo 'Downloading' $url$f ' ...'
    curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &

    wait # finish background tasks
    ```



## Convert Raw Data into Required Formats

Even though we have the raw images and annotations, we need to convert them into the required format
for most out of the box models. This is because when we load the data into our deep learning model,
it will expect the data to be in a certain format. 

The format is **largely determined** by how we write the `dataset` class in PyTorch (or TensorFlow).



## Label Tools 

### CVAT

#### macOS M1

Follow the [installation guide](https://opencv.github.io/cvat/docs/administration/basics/installation/):

1. Download Docker for Mac M1 (without Rosetta), as mentioned in [issue](https://github.com/docker/for-mac/issues/6232) opened on GitHub, we need to select **use docker compose v2** in the settings of docker as it is unselected by default, causing an error when using `docker compose`.
2. Download and install CVAT:
    
    ```bash
    $ git clone https://github.com/opencv/cvat
    $ cd cvat
    $ CVAT_VERSION=dev docker-compose pull
    $ CVAT_VERSION=dev docker-compose up -d
    ```
    
    Note that the commands above are **unique to macOS M1** as the CVAT installation guide does not fully support M1 yet, so have to go through some loops and hoops.
    
    See [issue](https://github.com/opencv/cvat/issues/4816) here for reference.
    
3. You can register a user but by default it will not have rights even to view list of tasks. Thus you should create a superuser. A superuser can use an admin panel to assign correct groups to other users. Please use the command below:
    
    ```bash
    docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
    ```
    
    username: django/django2
    
    password: ZGRu3pUsCw66HGa
    
4. Somehow you need to re-run steps 2-3 when you restart you computer?
5. See my short video clip for annotation steps.

#### Windows

1. Install WSL2

    Newer windows can directly call the following command in **adminstrator** Powershell:

    ```bash
    $ wsl --install
    ```

    If not, the safe choice is to follow the guide [here](https://docs.microsoft.com/en-gb/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package), starting from step 4.

2. Install Docker

    Download and install docker **[here](https://docs.docker.com/desktop/install/windows-install/)**.

3. Install Git

    Make sure there is Git for Windows.

4. Run CVAT

    1. Clone CVAT to local machine.
        
        ```bash
        $ git clone https://github.com/opencv/cvat
        $ cd cvat
        ```
        
    2. Run docker containers.
        
        ```bash
        $ docker-compose up -d
        # run below since now latest version has issues
        $ docker-compose -f docker-compose.yml -f docker-compose.dev.yml build
        ```
        
        However, there is an issue opened just 2 weeks ago:
        
        - See https://github.com/opencv/cvat/issues/4888 and https://github.com/opencv/cvat/issues/4816
    3. Create username and password:
        
        ```bash
        # enter docker image first
        $ docker exec -it cvat_server /bin/bash
        # then run
        $ python3 ~/manage.py createsuperuser
        ```
        
        username: django
        
        password: ZGRu3pUsCw66HGa
    
5. Open the installed Google Chrome browser and go to [localhost:8080](http://localhost:8080/). Type your login/password for the superuser on the login page and press the *Login* button. Now you should be able to create a new annotation task. Please read the [CVAT manual](https://opencv.github.io/cvat/docs/manual/) for more details.

---

## End-to-End ML Workflow

### Clarify the Problem and Constrains

This one can ask Lucas more when the time comes, we will assume that this is a 
real life problem that requires a solution.

We can then ask the below potential questions.

- What is the problem?
- How do end users use the product and benefit from it?
- Benefit Structure?
- Ethical issues (e.g. discrimination, privacy, etc.)
- Deployment: latency vs accuracy etc;
- Where we deploy the model on?
- ...... etc

We can then form a problem statement along with a summary, where we will constantly 
refer to it as we move along.

### Establish Metrics

Once we are clear on the problem statement, we can define our metrics:

**Model Performance Metrics**

- The typical metrics such as accuracy, precision, recall, F1 score, etc.
- More sophisticated usage includes AUC, ROC, etc.
- Constrainment includes $\max \textbf{Mean Average Precision}@\textbf{Mean Recall} 0.8$
    - which means you constrain mean recall to be at least 0.8 while maximizing mean average precision.
- Model Performance is not everything to stakeholder, sometimes may need to consider their KPIs.

**Business Performance Metrics**

- Unsure, need some product sense, but I guess our project is not too clear yet.

### Data, Data, Data

We can further split this into 3 parts:

- Data Collection
- Data Cleaning
- Data Preparation

!!! info
    Arguably, the most important part of the whole process is the data, as we move from a **model-centric**
    approach to a **data-centric** approach (re: Andrew NG), we really need to ensure the quality of the data.

#### Data Collection

Place holder.

#### Data Cleaning
      
Place holder.
            

#### Data Preparation

TODO:

- To be refined, here is some collection of my initial findings.
- To test on coco128 when developing next time.

- There are no bounding box annotations, therefore we need to label/annotate the images:
    - Created a branch `node-convert-to-coco` to convert the node's output data to coco format.
        - This branch will make inference on the images and give us the bounding box annotations.
        - This reduces overhead for my quick prototyping, but is not the best if one cares about 
          the quality of data. This is because although inferencing on human is easy, it is not
          trivial when the human wears a **helmet**, the bounding box is not tight above. 
        - The output using `csv_writer` will be converted to COCO.
        - See my tmp.py for more details.
    - Once converted, tested it on google colab to see if it works. It did since macOS cannot run cuda
      and this repo only allows cuda for training, we can loosen this restriction.  
      See my google drive.

- See YOLOXX to see how to annotate using COCO format, this was developed on macOS initially.

!!! Danger "TODO"
    Some considerations:

    - Do we need a node to convert our inference results to COCO/VOC format? We can have a generic node 
      for this purpose; like we enforce a certain format in the form of csv/df, then convert them to either.  

##### COCO Dataset

- [COCO Detection Eval](https://cocodataset.org/#detection-eval)



##### Pascal VOC Dataset

TODOS:

- Currently convert coco to voc online, need write a script maybe since I wrote one for coco temporarily.

For YOLOX, Pascal VOC is not well maintained, so for now the folder structure is rigidly defined below:

```tree
YOLOX/
├── datasets/
│   ├── VOCdevkit/
│   │   ├── VOC2012/
│   │   │   ├── sp_ppe_voc_helmet_mask_vest/
│   │   │   │   ├── Annotations/
│   │   │   │   ├── ImageSets/
|   |   |   |   |   ├── Main/
|   |   |   |   |   |   ├── train.txt
|   |   |   |   |   |   ├── val.txt
|   |   |   |   |   |   ├── test.txt
|   |   |   |   |   |   ├── trainval.txt
│   │   │   │   ├── JPEGImages/
```

Without changing source code, the `VOCdevkit` and `VOC{year}` folder names are fixed and not negotiable.
I attempted to do away with the `VOC{year}` folder, but it really requires changing a few places, and
I am aware that even PyTorch's own `voc.py` dataset uses this same format. So I keep it as such for now
but it is possible to loosen this restriction.

- **Annotations** folder contains all the xml files, which is each image's annotations info in Pascal VOC format.
- **JPEGImages** folder contains all the images.
- **ImageSets** folder contains the train/val/test split, which is a txt file with each line being the image name
  without the extension. For example, if the image is `000001.jpg`, then the line in the txt file is `000001`.
    - Note we do not need to use `trainval.txt` since we can just use `train.txt` and `val.txt` to train and validate.
    - Note we should find a way to use KFolds here, might be cumbersome to train every fold manually?

### Model Training

- [Training your own YoloX Object Detection Model on Colab - YoloX Object Detection Model Deployment](https://www.youtube.com/watch?v=be_D3V9Pxlg&t=2371s)
    - This quite useful as a starter.
- Coding your own dataset or dataloader class
    - [MMDetection Dataset Class](https://github.com/open-mmlab/mmdetection/tree/master/mmdet/datasets)
    - https://github.com/pytorch/vision/blob/main/torchvision/datasets/voc.py

#### Config (Exp) File

The main config file of YOLOX resides in the `exps/` folder.

```python title="YOLOX/exps/default/yolox_s.py" linenums="1"
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = NUM_CLASSES  # 20
        print("num_classes", self.num_classes)

        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
```


## Steps

1. Make the data into pascal voc format and place it under `YOLOX/datasets/VOCdevkit` folder. Note you need to create `VOCdevkit` folder first.
   1. Follow exactly the format I have inside my folder.
2. When making the data, I hardcoded a bit, make sure eventually is same format as the one here for now https://www.youtube.com/watch?v=be_D3V9Pxlg&t=2371s with some tweaks.
3. Download weights to weights folder `YOLOX/weights`
   
   ```bash
   mkdir weights
   wget.exe -P .\weights\ https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth
   ```
4. Create `exp` file, meaning the configuration file for the experiment.
   
   ```bash
   mkdir exp/custom/sp_ppe
   cd exp/custom/sp_ppe
   touch sp_ppe_voc.py
   cd ../../../
   ```

   see code inside, this file is very important as it overwrites the base class.

4.1. Note it is very important to change coco class and voc class file.

5. Run the training
   
   ```bash
   python tools/train.py -f .\exps\custom\sp_ppe\sp_ppe_voc_human_only.py -d 1 -b 4 --fp16 -o -c .\weights\yolox_s.pth
   ```

   if there is Weights & Biases:

   ```bash
   python tools/train.py -f .\exps\custom\sp_ppe\sp_ppe_voc_human_only.py -d 1 -b 4 --fp16 -o -c .\weights\yolox_s.pth \
   --logger wandb \
   wandb-project yolox \
   wandb-log_checkpoints True \
   wandb-num_eval_images 3 \
   ```

   where `num_eval_images` means at every evaluation step, the dashboard also shows 3 images from the validation set along with the predicted bounding box.

6. Run test
   

   ```bash
   MODEL_PATH = 'C:\Users\reighns\reighns_ml\ml_projects\YOLOX\YOLOX_outputs\sp_ppe_voc_helmet_mask_vest\latest_ckpt.pth'
   TEST_IMAGE_PATH

   python tools/demo.py image -f .\exps\custom\sp_ppe\sp_ppe_voc_helmet_mask_vest.py -c .\YOLOX_outputs\sp_ppe_voc_helmet_mask_vest\last_epoch_ckpt.pth --path .\datasets\VOCdevkit\VOC2012\sp_ppe_voc_helmet_mask_vest\test\helmet--147-_jpg.rf.ed1d0dab4fc4b4f0c4213e6569a7ef02.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
   ```

## Weights & Biases

See https://docs.wandb.ai/quickstart for more details. But basically use

```bash
pip install wandb
wandb login
```

however for now comment out a chunk of code in logger since it has `self.cats` which only works for coco.

## Model Capability

We first check if model can memorize, then generalize, the former is important to check for bugs, and to ensure our model has basic capability to learn, but no way does it 
guarantee that it can generalize. Regardless, we still want to ensure this as if it cannot even memorize in-sample data points, it for sure cannot generalize.

## References


- https://docs.wandb.ai/guides/track/log/media to log bounding boxes
- https://www.youtube.com/watch?v=be_D3V9Pxlg&t=2371s
- https://github.com/awsaf49/bbox/blob/main/bbox/utils.py
- https://www.kaggle.com/code/ayuraj/train-yolov5-cross-validation-ensemble-w-b/notebook
- https://github.com/roboflow-ai/YOLOX
- https://www.kaggle.com/code/awsaf49/great-barrier-reef-yolov5-train/notebook#%F0%9F%93%81-Create-Folds
- Transfer learning for YOLOX: https://github.com/Megvii-BaseDetection/YOLOX/issues/1105 + https://github.com/Megvii-BaseDetection/YOLOX/pull/1156
- Converting darknet or yolov5 datasets to COCO format for YOLOX: YOLO2COCO [from Daniel](https://github.com/RapidAI/YOLO2COCO)
* https://stackoverflow.com/questions/67733876/create-pascol-voc-xml-from-csv
https://pyimagesearch.com/2022/05/02/mean-average-precision-map-using-the-coco-evaluator/
https://wandb.ai/manan-goel/yolox-nano/reports/Tracking-your-YOLOX-Runs-with-Weights-Biases---VmlldzoxNzc0NjA0
https://docs.wandb.ai/guides/integrations/other/yolox
https://www.kaggle.com/code/remekkinas/yolox-training-pipeline-cots-dataset-lb-0-507/notebook#6.-RUN-INFERENCE
https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
https://blog.roboflow.com/how-to-train-yolox-on-a-custom-dataset/
https://www.kaggle.com/code/litaldavar/hard-head-detection-with-yolov5/notebook
https://www.kaggle.com/code/billiemage/object-detection
# Human_Detector_v2 by Abhishek Saharia
## System Design
The system communicates with several modules using OOP concepts to provide the best possible processing speed and accuracy while enabling easy future development.

![system design (Reload if image not visible)](/AI/Abhishek/Human_Detector_v2.3/system_design.png "Human_Detector_v2 by Abhishek Saharia")
## Input Parameters
While running the detection system, various parameters can be passed as input arguments through the CLI.

```file-path``` : The location of the video or image files to run the detection on. The system auto detects whether the file is an image or a video.

```gpu-num``` : The number of GPUs to run the system on. For PCs with multiple GPUs, it can run parallelly on all GPUs for faster processing. 

```include-score``` : Whether to include the confidence scores in the output file.

```threshold``` : Can be set to a *float* between 0.0 and 1.0 to specify a custom confidence score threshold for human detection.

```position-threshold``` : To specify a custom confidence score threshold for position estimation.

```skip-scaling``` : Whether to skip the scaling part and only conduct detection<sup>1</sup>.

1 *Foot position generated wont consider occluded foot.*
## Output Format
The output is obtained as an ndjson file containing data in the following format:
*For a video file containing **n** frames*
```
{"frame_no": 0, "people": [[x11,y11,x21,y21,xf1,yf1,cs11,cs21],.....,[x1k,y1k,x2k,y2k,xfk,yfk,cs1k,cs2k]]}
{"frame_no": 1, "people": [[x11,y11,x21,y21,xf1,yf1,cs11,cs21],.....,[x1k,y1k,x2k,y2k,xfk,yfk,cs1k,cs2k]]}
.
.
.
{"frame_no": n, "people": [[x11,y11,x21,y21,xf1,yf1,cs11,cs21],.....,[x1k,y1k,x2k,y2k,xfk,yfk,cs1k,cs2k]]}
```

For a Bounding Box k,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**(x1k,y1k)** is the **top left coordinate** of the bbox.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**(x2k,y2k)** is the **bottom right coordinate** of the bbox.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**(xfk,yfk)** is the **foot keypoint coordinate**.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**cs1k** is the **confidence score of the bbox**<sup>2</sup>.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**cs2k** is the **confidence score of the foot keypoints**<sup>2</sup>.

2 *inclusion of confidence score needs to be specified as a CLI argument.*

## Steps to install and run:
1. Create a virtual environment.
    1. Linux:
        ```
        python3 -m venv <directory>/venv
        ```
    2. Windows:
        ```
        python -m venv <directory>\venv
        ```
2. Go to venv directory.
    ```
    cd <directory>
    ```
3. Activate the environment.
     1. Linux:
        ```
        source venv/bin/activate
        ```
    2. Windows:
        ```
        ./venv/Scripts/activate
        ```

4. Go to project directory and install: 
    ```
    pip install -r requirements.txt
    ```
5. Download the model file with initialised weights and save it in the **weights** folder: [yolo4_weight.h5](https://drive.google.com/file/d/11Xbh0PQG1uU_qiIM0YaudaAQ01_rcm0z/view?usp=sharing) 
6. Run human detector 
    ```
    python run_detector.py --file-path <input video or image file path> --gpu-num <number of gpus> --include-score <flag> --threshold <threshold of confidence score> --position-threshold <threshold of confidence score> --skip-scaling <flag>
    ```
    
7. output ndjson file is saved in **output** folder
8. Default values ```gpu-num=1```,```threshold=0.4```,```position-threshold=0.11```.

## Version 2.3 update:
* Added **Occlusion Detector**, a newly trained ANN model to filter out occluded bboxes.
* Scaling uses the trained Occlusion detector along with a dynamic ANN model using **ensemble learning**.
* Added a **Scaling** method to scale the occluded bboxes to foot location based on type of occlusion and generate foot keypoints.
* Added a new script to combine Human Detection and Scaling together into a highly connected system.
* Optimized VRAM usage to prevent unnecessary allocation of resources.
* Added data **cache** feature to store intermediate data.
* Added feature to quit detection at anytime and carry on with scaling on the processed part of the video instead of aborting.
* Added a **progress counter** for the scaling process.
* Added support for setting a custom score threshold for position estimation.
* Improved **performance** and resolved various bugs, exceptions and conflicts.
* The following modules were added/made changes to:
    1. ```occlusion_filter.py``` (added)
    2. ```foot_detector.py``` (added)
    3. ```MoveNet_core.py``` (added)
    4. ```draw_keypoints.py``` (added)
    5. ```foot_keypoint_writer.py``` (added)
    6. ```imputR.py``` (added)
    7. ```run_detector.py``` (added)
    8. ```json_writer.py``` (modified)
    9. ```human_detector_v2.py``` (modified)
    10. ```json_reader.py``` (modified)
    11. ```foot_predictor.py``` (removed)

## Version 2.2 update:
* Added **Multi GPU** support to run parallelly on multiple GPUs. It can be passed as CLI argument
* Added compatibility with **headless opencv**
* Made changes to generate output ndjson file in a new format
* Modified the **fps counter** update on a single line
* Added support for use in **Linux environment**
* Custom confidence score threshold and option to include confidence scores in output
* Added support for detecting **images**. Automatically decides whether the input file is image or video
* The following modules were added/made changes to:
    1. ```yolov4_core.py``` (modified)
    2. ```json_writer.py``` (modified)
    3. ```human_detector_v2.py``` (modified)


## Version 2.1 update:
* Added **Hybrid imputation** to scale the bboxes to the feet location.
* Updated ```system_design.png```
* Added counter for processing time and post processing time in *hh:mm:ss* format
* The following modules were added/made changes to:
    1. ```foot_predictor.py```
    2. ```json_writer.py``` (modified)
    3. ```human_detector_v2.py``` (modified)

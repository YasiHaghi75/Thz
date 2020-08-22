# Critical Object Detection in Terahertz Image Sequences
This repository contains the python codes and image datasets used in the project **Critical Object Detection in Terahertz Image Sequences**. 

## Terahertz Imaging
Recently millimeter-wave imaging (Terahertz Imaging) has been very popular among different applications. An example of these applications is security check at public and crowded places like airports and train stations.
**Terahertz** radiations can penetrate through large variety of materials and are less harmful for human body compared to X-ray radiations. Also metals highly reflect Terahertz radiations. These characteristics have made Terahertz imaging a suitable method to detect concealed metallic objects. 

## Methodology
There are two methods of THz imaging; 
  1. **active-mode**
  2. **passive-mode**

In this research **active mode** is used in which terahertz radiation of the external source is detected after back scattering from the object, therefore there is always a possibility for a metallic surface not to face any radiations as it is highly dependent on the angle between the radiations and metallic surface. This problem shows up in active terahertz imaging and therefore there are some frames containing metallic objects with no reflections. There is also a chance of reflections from human body organs in frames without any metallic objects.
To overcome this problem, we have taken a **sequential** approach. We use a sequence of terahertz images to determine whether or not a person has critical metallic objects with him. As the detection of critical objects should be done in a very short time, we have used simple linear machine learning models, specifically **SVM**, to do the classification.

## Usage
1. Create a Python environment and run `pip install -r requirements.txt`.
2. To test the max_mean_var model on your own experiment run test_max_mean_var.py as below:
```
python ./test_max_mean_var.py --model <path to the trained model> --scalar <path to the trained scalar> --trial <path to trial images directory>

```

## Dataset
The dataset used in this project is obtained using [Terasense](https://terasense.com/) body scanner system. Subjects (with or without metallic object) were asked to walk from the distance of 3 meters (from camera lens) towards the scanner and their movement had been recording meanwhile. The output THz sequence of images is used to train and test the model.
Below there is a sample of output THz image from a metallic object (right) and its corresponding visible light image in RGB (left). 

![dataset sample](https://github.com/YasiHaghi75/Thz/blob/master/md_images/dataset1.png?raw=true)

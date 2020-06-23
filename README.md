# Thz
- Recently millimeter-wave imaging has got very popular among different applications.
An example of these applications is security check at public and crowded places like airports and train stations.
- **Terahertz** radiations can penetrate through large variety of materials and are less harmful for human body compared to X-ray radiations. 
Also metals highly reflect Terahertz radiations. 
These characteristics have made Terahertz imaging a suitable method to detect concealed metallic objects. 
- There are two methods of THz imaging; 
  1. active-mode 
  1. passive-mode 
- In this research **active mode** is used in which images are taken from the reflection of THz radiations from different surfaces, therefore there is always a possibility for a metallic surface not to face any radiations as it’s highly dependent on the angle between the radiations and metallic surface. This problem shows up in active Terahertz imaging in which Terahertz radiation of the external source is detected after back scattering from the object, so there are some frames containing metallic objects with no reflections. 
There is also a chance of reflections from human body organs in frames without any metallic objects.
- To overcome this problem, we have taken a **sequential** approach. 
- We use a sequence of Terahertz images to determine whether or not a person has critical metallic objects with him. 
- As detection of critical objects should be done in a very short time, we have used simple linear machine learning models, specifically **SVM**, to do the classification.   


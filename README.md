# Event-Driven Figure-Ground organisation model for the humanoid robot iCub 

Correspondent author: 
Giulia D'Angelo
- Istituto Italiano di Tecnologia; email: giulia.dangelo.1990@gmail.com
- Currently Marie Skłodowska-Curie Postdoctoral Fellow at The Czech Technical University in Prague; email:  giulia.dangelo@fel.cvut.cz


Figure-ground organisation is a mechanism of perceptual grouping that serves the detection of objects and their boundaries, needed by any agent to interact with the environment.
Current approaches to figure-ground segmentation rely on classical computer vision or deep learning, requiring vast computational resources, in particular during the training phase of the latter. 
Inspired by the primate visual system, we developed a bio-inspired perception system to compute figure-ground organisation for the neuromorphic robot iCub. 
The proposed model distinguishes foreground objects from the background, thanks to a hierarchical biologically plausible architecture, exploiting event-driven vision. 
Differing from classical approaches, the use of event-driven cameras 
reduces data redundancy and computation. 
The system has been qualitatively and quantitatively assessed both in simulation and using the event-driven cameras mounted on the iCub looking at different scenarios and stimuli. The model successfully segments items in diverse real-world scenarios, showing comparable results against its frame-based version on simple stimuli and on the Berkeley Segmentation dataset benchmark [1]. This model further offers a valuable addition to hybrid systems, complementing conventional computer vision deep learning models by processing only relevant data in detected Regions of Interest (ROI), thereby enabling low-latency autonomous robotic applications without redundant information.

[1]: David Martin, Charless Fowlkes, Doron Tal, and Jitendra Malik. A database of human segmented natural432
images and its application to evaluating segmentation algorithms and measuring ecological statistics. In433
Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001, volume 2, pages 416–423. IEEE,434
2001.


![Figure ground organisation](https://github.com/event-driven-robotics/figure-ground-organisation/blob/main/EDFG.png)

This work is based on previous publications for bioinspired visual attention mechanisms for the humanoid robot iCub. 

- [Proto-object based saliency for event-driven cameras](https://ieeexplore.ieee.org/abstract/document/8967943)
- [Event driven bio-inspired attentive system for the iCub humanoid robot on SpiNNaker](https://iopscience.iop.org/article/10.1088/2634-4386/ac6b50/meta)
- [Event-driven proto-object based saliency in 3D space to attract a robot’s attention](https://www.nature.com/articles/s41598-022-11723-6)


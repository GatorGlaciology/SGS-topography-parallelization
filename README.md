<img src="https://raw.githubusercontent.com/GatorGlaciology/GStatSim/main/images/GatorGlaciologyLogo-01.jpg" width="100" align= "right">

# SGS Topography Parallelization

A parallelized Sequential Gaussian Simulation (SGS) script in Python designed for simulating subglacial topography. It is modeled after the
non-stationary SGS with adaptive partitioning technique presented in the open source package [GStatSim](https://github.com/GatorGlaciology/GStatSim).
Our script is intended for geostaticians interested in reducing the time required to generate subglacial topographic realizations. 
Using our parallelization strategy, have noticed a speed improvement of up to 14x when compared with serial implementations of SGS.
Additionally, we created a Jupyter notebook that walks through the steps of our parallelized algorithm. This notebook features an interactive 3D 
visualization of the simulated bed topography using [PyVista](https://github.com/pyvista/pyvista).

We hope our script will make topographic modeling with SGS more accessible and time efficient.
If you have any feedback or suggestions, feel free contact me at [nathanschoedl@ufl.edu](nathanschoedl@ufl.edu). 

![Github_p2](https://user-images.githubusercontent.com/73554694/215893567-e631e438-ca84-44b2-98ab-ff98ec079d22.png)


## Python Script

This script enables users to generate multiple topographic realizations at a specified resolution using subglacial bed-elevation datasets. 

### Files required to execute script:

* **sgs_main.py** - Main file, provides an overview of simulation steps
* **sgs_preprocess.py** - Functions to prepare data for simulation 
* **sgs_alg.py** - Functions that perform calculations to simulate bed elevation in parallel 
* **sgs_plts.py** - Functions to plot topographic realizations

### How to use?

1. Ensure all the dependencies are already installed on your system
    * All dependencies can be installed using *pip install \[package\]* 
2. Download this repository with *git clone 
[https://github.com/GatorGlaciology/SGS-topography-parallelization](https://github.com/GatorGlaciology/SGS-topography-parallelization)*
3. Enter *python3 sgs_main.py*
4. Follow the command line arguments 
    * Enter relative or absolute file path if dataset is not in same folder as **sgs_main.py**
    * Dataset must be a csv file in polar stereographic coordinates 
5. Find the resulting csv file(s) and plot(s) in the **Output** folder

## Notebook 

**SGS_parallel_demo.ipynb** is designed to demonstrate the steps of SGS and increase understanding on how SGS was parallelized. Additionally, users can
input their own dataset to generate results. To run, please ensure all dependencies are installed. Using PyVista, after simulating the bed elevation 
values, we can visualize the topography as shown below:

![Final_readme](https://user-images.githubusercontent.com/73554694/217956514-0386089a-c404-4a22-be4c-63add0fc980a.gif)

## Dependencies

* **numpy**
* **pandas**
* **skgstat**
* **multiprocessing**
* **sklearn**
* **matplotlib**
* **itertools**
* **pyvista**
   * Only required for Jupyter Notebook
   * **ipyvtklink** required for interactive visualization

## Contributors 

Nathan Schoedl, University of Florida

(Emma) Mickey MacKie, University of Florida

Michael Field, University of Florida

Eric Stubbs, University of Florida

Allan Zhang, University of Florida

Matthew Hibbs, University of Florida

Mathieu Gravey, Utrecht University

# SPEAR: a Simple Python Environment for statistical estimation of Adaptation and Reliability

SPEAR is a simple Python tool that permits estimating the distance between two systems. 

Each system consists of three distinct components: 
  * a *process* describing the behaviour of the program; 
  * a *data space*; 
  * an *environment evolution* describing the effect of the environment on the data space.

Two systems are compared via a (pseudo)metric, called the *evolution metric*. Thanks to the possibility of extrapolating process behaviour from that of the system typical of our model, this metric allows us to
  * verify how well a program is fulfilling its tasks by comparing it with its specification;
  * compare the activity of different programs in the same environment;
  * compare the behaviour of one program with respect to different environments and changes in the initial %environmental 
conditions.

Via the metric we introduce some dependability properties of a program, called *adaptability* and *reliability*.

Adaptability denotes the ability of the program to lead the system back to its desired behaviour, within a given time horizon, when some perturbation is applied to the initial data space. Reliability is the untimed version of adaptability.

In [temperature.py](./temperature.py) we show how SPEAR can be used to model a simple heating system in which the thermostat, the program, has to keep the temperature of the room within a desired comfort interval. We use our algorithm to evaluate the differences between two heating systems having the same program but starting from different initial conditions. Finally, we apply it to study the adaptability and reliability of the considered program

## Download 

To download SPEAR you have just to clone GitHub project:

```
git clone https://github.com/quasylab/spear.git
```

Run this command in the folder where you want to download the tool.

## How to run experiments

To run experiments Python3 is needed. Moreover, the following Python packages must be available:
  * [numpy](https://numpy.org) >= 1.18.4
  * [scipy](https://www.scipy.org) >= 1.4.1
  * [matplotlib](https://matplotlib.org) >= 3.0.3
  
To install all the required packages you can just run:

```
pip install -r requirements.txt
```

If all the needed package are installed in your system, you have to execute:

```
python temperature.py
```

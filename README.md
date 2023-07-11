# Readme.md

<p align="center">
  <img src="pic/picture" alt="banner"/>
</p>

<p align="center">
  Introduction of an easy method to control Franka in Isaac sim
</p>


## Contents

- [Description](#Description)
- [How to install](#How-to-install)
- [How to use](#How-to-use)


## Description
<a name="Description"/>

<p align="justify">
  Introduction of an easy method to control Franka in Isaac sim with inverse kinematic.
</p>


## How to install
<a name="How-to-install"/>

#### Requirements
* You must have a computer compatible with Isaac Sim 2021.2.1, please check the [official documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html).
* You must install proper GPU driver, Isaac Sim and Orbit. Please check [orbit documentation](https://isaac-orbit.github.io/orbit/) to install orbit.

Please do not use ```sudo apt-get install nvidia-***``` to install GPU driver. Download ```.run``` file from [Nvidia official](https://www.nvidia.cn/Download/index.aspx?lang=cn) and blacklist nouveau first.

#### Steps
 1. Download this Git.
 2. Copy Franka-Isaac-Sim to ~/Franka-Isaac-Sim
 3. Install your orbit library to ~/Orbit, where orbit.sh locate.


## How to use
<a name="How-to-use"/>

Open a terminal and run the following commands:

#### To run
 1. cd ~/Franka-Isaac-Sim
 2. ~/Orbit/orbit.sh -p franka.py

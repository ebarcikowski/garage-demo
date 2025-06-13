# Garage Demo

A minimal demo version of my (silly) garage door detector. See 
https://github.com/ebarcikowski/garage for more info.

## Overview

This is basic end to end example of creating, training and deploying a 
Keras ML model. This is mostly for fun but does actually work with 
a home camera in my garage, last I tried anyway.

This repository serves a working example and is designed to go through
the code more than running the programs. However, code examples are most
effective when they do something.

## To Run

This requires Python between 3.9 and 3.12 for tensorflow. Everything below
needs to be run in the root directory of the project.  

Create an environment however you prefer. For example,

```sh
python -m venv venv
```

Then, activate

```sh
source venv/bin/activate
```

Install the requirements

```sh
pip install -r requirements.txt
```

Run the training

```sh
python train.py
```

Run the testing

```sh
python infer.py
```


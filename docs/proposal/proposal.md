# School of Computing — Year 4 Project Proposal Form

```
Edit (then commit and push) this document to complete your proposal form..
```

```
Do not rename this file.
```

## SECTION A

#### Project Title: Analysing the accuracy of several different gap-filling techniques on real-world climate datasets..

#### Student 1 Name: Kevin Boyle

#### Student 1 ID: 19731615

#### Student 2 Name: Mark Reid

#### Student 2 ID: 19414892

#### Project Supervisor: Mark Roantree

```
Ensure that the Supervisor formally agrees to supervise your project; this is only recognized once the Supervisor assigns
herself/himself via the project Dashboard.
```

```
Project proposals without an assigned Supervisor will not be accepted for presentation to the ApprovalPanel.
```

## SECTION B

### Introduction

The goal of our fourth-year project is to develop a system in which climate scientists are able to provide time-series datasets
with gaps throughout and be returned gap-filled data using multiple interpolation techniques. We will provide the user with a
detailed explanation as to why and when each interpolation technique is most appropriate and effective. We hope that our
system will benefit those in the world of climate/weather research, as they work with real-world data which is known to be
plagued with gaps throughout. The user will be able to choose a dataset, each of which contains a different level of
sensitivity (size of the gaps present).

### Outline

For this project, the overwhelming majority of the work will be based on the processing of data. We will introduce the gaps
from several datasets which, as mentioned before, will solely focus on climate/weather using scientific methods of gap
insertion. The gaps will not be inserted at random. We must implement the different techniques of gap-filling which we plan on
using Python to develop. The ideal user will be a climate scientist who regularly works with real-world datasets as these
oftentimes contain many gaps. There will be no fancy GUI, rather just the examination of the resulting data.

### Background

The idea came to us through our studies of CA4010 Data Warehousing & Data Mining. We both developed an interest in
machine learning and felt that it would be intriguing to base our fourth-year project on a related topic. We have decided to
focus our project on comparing several different interpolation methods for filling gaps in large datasets. These datasets will
contain real-world time-series data focusing on weather/climate. Real-world data is oftentimes plagued with gaps, due to
sensors or the connection between sensors failing.

We believe that this is applicable as the majority of statistical methods and environmental models, require serially complete
data, so gap filling is a routine procedure in the preprocessing of datasets. Our target user is Climate scientists as they
consistently work with real-world data.

### Achievements

The presence of missing data is a challenging issue in processing real-world datasets. The aim of our fourth-year project is to
compare the effects several different techniques of gap-filling have on a large number of weather/climate datasets.

Ideally, our system will be able to allow a user to input a dataset that contains real-world, time-series data taken from
sensors at several weather stations. Datasets containing climatic variables are frequently made up of time-series data
covering different time spans and are plagued with data gaps.

We then will offer the user several different gap-filling techniques which they can choose from to be applied to the dataset. We
aim to focus on several _interpolation methods_. The user will also be presented with a detailed explanation of the functionality
of each gap-filling technique and where each is most applicable. After the user selects a method, our system will then apply
the gap-filling function on the dataset and return it with the gaps filled.

### Justification

With the current circumstances regarding climate uncertainty, we found that now could be a great opportunity to focus on the topic of climate. Furthermore, machine learning is a rapidly growing field on that we would like to focus our project on. Given our previous experience in Data Warehousing and Data Mining, we would like to touch on what we learned and expand past this, focusing heavily on the machine-learning side of data science.

### Programming language(s)

- Python
- SQL

### Programming tools / Tech stack

```
● Interpolation tools available inside SciPy.
● Python to develop our own interpolation methods which will be compared with SciPy’s.
● SQL to facilitate our database (Containing the different datasets).
● Jupyter Notebook (compile aspects of our project in one place).
```

### Hardware

```
● No hardware is required other than a machine with access to the internet.
```

### Learning Challenges

```
● Machine Learning
● Research of methods of Interpolation
● Preprocessing of data - insertion of gaps inside each dataset.
● Identifying what techniques of gap filling are most effective in each scenario/database.
● Implementing each method of gap-filling.
```

### The breakdown of the work

#### Student 1: Kevin Boyle

```
● Research and Implement the insertion of gaps in each dataset/preprocessing our Datasets.
● Implementation of 50% of our gap-filling techniques.
● Design of Graphical user Interface.
```

#### Student 2: Mark Reid

```
● Analysing the efficiency of each gap-filling method and where each is most appropriate.
● Implementation of 50% of our gap-filling techniques.
● Developing the Graphical User Interface & Content.
```

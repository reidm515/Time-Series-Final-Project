# School of Computing — Year 4 Project Proposal Form

```
Edit (then commit and push) this document to complete your proposal form..
```

```
Do not rename this file.
```

## SECTION A

#### Project Title: An Intelligent Allergy Assistant

#### Student 1 Name: Kevin Boyle

#### Student 1 ID: 19731615

#### Student 2 Name: Mark Reid

#### Student 2 ID: 19414892

#### Project

#### Supervisor:

#### Mark Roantree

```
Ensure that the Supervisor formally agrees to supervise your project; this is only recognised once the Supervisor assigns
herself/himself via the project Dashboard.
```

```
Project proposals without an assigned Supervisor will not be accepted for presentation to the ApprovalPanel.
```

## SECTION B

### Introduction

The goal of our fourth year project is to develop an application with the aim to aid individuals with allergens.

We aim to develop an application which will give users the opportunity to scan and automatically detect allergens inside
products using it’s barcode. Additionally, we believe it would be beneficial to include a feature where users can record their
own allergies and flag instances where these have been detected in a product, or give warnings if a scanned product contains
their specific allergens.

### Outline

For this project, we plan on creating a mobile application in Python, using Pycharm and Vscode to facilitate our python code.
The majority of our applications frontend will be developed using Flutter, a Python Framework which is used to develop and
design mobile applications with a natural user interface. We chose to use Flutter as it is very versatile and easy to use, it
supports six platform targets: Android, iOS, Windows, macOS, Linux, and web applications.

We plan on using AWS Lambda to assist in creating the backend of our mobile application. Lambda is a “serverless,
event-driven compute service that lets you run code for virtually any type of application or backend service without
provisioning or managing servers”. It provides an HTTP API for custom runtimes tor receive invoked events and sends
response data back within it’s own environment. Using these technologies, we aim to implement an application that allows the
user to scan a food item with their camera to check for any allergic ingredients.

Additionally, to integrate our Frontend and Backend together, we will be using Dart which comes with Flutter when it is
installed.

### Background

Initially, the idea came after we both searched through numerous lecturer’s project ideas. We agreed that doing something
that was related to the two of us would make our project more intriguing. As we both suffer from food allergies, we felt that
this idea not only could benefit the pair of us, but the large numbers of people across Ireland that who suffer with food
allergens, some of them being life-threatening.

We both believe that in Ireland, it can be difficult for consumers to read the ingredients on every product. This is mostly due to
the location on the product or the size of the font, hence it is a tedious shopping experience. We feel that our application could
help with this and make grocery shopping a safer and more enjoyable experience.

### Achievements

Ideally the function of our application would be for any user to scan a product in their local supermarket and in an instant be
warned if the product contains allergic ingredients. We have found a similar feature inside the MyFitnessPal application, which
is available on the Google Playstore.

MyFitnessPal offers users the ability to scan a products barcode and be shown the amount of calories it contains and that
specific products macro nutrients. We believe that we could implement something similar to this, but instead of returning a list
of macro nutrients we would return the ingredients that are flagged as allergens and warn the user if others have experienced
negative reactions after consuming that product.

### Justification

Webothbelievethatthisprojectcouldbeusefulinmanydifferentways.It’sestimatedthat26%ofthepopulationsufferwith
allergyconditionsnationwideandover10%worldwide.Asbothofusexperienceallergiesourselves,weknowfirsthandthe
feelingofuncertaintywhenpurchasingproductsinstore.Wealsoknowthefeelingofconfusionwhenreadingthebackof
products,notknowingwheretheallergicingredientsarelisted.Webelieveitwouldbeabenefittoindividualswithallergiesif
theyhadtheabilitytoscanandautomaticallydetectallergensfromingredientslistings,recordtheirindividualallergensand
flaginstanceswherethesehavebeendetectedinaproductandincludeafeaturewhereuserscanreportallergiceventsand
this is analysed for associations with various food products.

Inouropinion,providingthisservicewouldmakefoodshoppingalessconfusingandtediousexperienceforindividualswith
allergies, and help reduce the risks of consuming allergic ingredients especially as some allergies can be life-threating.

### Programming language(s)

- Python
- Flutter
- AWS Lambda
- Javascript
- Dart

### Programming tools / Tech stack

- We plan on using PyCharm and VSCode to facilitate the bulk of our code during this
  project.
- We will design and build the front-end of our application using Flutter.
- We will use AWS Lambda to create our serverless mobile application, putting our
  functions to separate its development lifecycle from that of front-end clients, which will
  make developing our mobile application less complex to develop and maintain.
- Dart to integrate the front-end of our application and the backend.
- We will be using DCU Computing Gitlab for our project repository.
- Over the course of the project, we will be using Trello to report, assign and review our work and progress.

### Hardware

- Andorid or IOS mobile phone with a back facing camera.

### Learning Challenges

```
● Building and testing the accuracy of our Predictive Model.
● Web scraping
● New Languages and libraries within Python, such as Jupyter notebook and Pandas within Python.
● Returning the output from the source-code inside of our web application.
```

### The breakdown of work

#### Student 1: Kevin Boyle

```
● Setup of UpciteMDB API.
● Design of GUI & Development of the Front-End of the application using Flutter.
● Testing - Unit Testing, Integration Testing and User Testing.
```

#### Student 2: Mark Reid

```
● Backend Work - AWS LAMBDA
● Database Fetching - Dart
● Testing - Functional & Regression Testing.
```

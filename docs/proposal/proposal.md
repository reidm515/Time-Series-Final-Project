School of Computing — Year 4 Project Proposal Form 

Edit (then commit and push) this document to complete your proposal form.. 

Do not rename this file. 

SECTION A 


|||
| :- | :- |
|Project Title: |Machine Learning - Building a Predictive Model.|
|Student 1 Name: |Kevin Boyle|
|Student 1 ID: |19731615|
|Student 2 Name: |Mark Reid|
|Student 2 ID: |19414892|
|<p>Project </p><p>Supervisor:</p>|Mark Roantree|


Ensure that the Supervisor formally agrees to supervise your project; this is only recognised once the Supervisor assigns herself/himself via the project Dashboard. 

Project proposals without an assigned Supervisor will not be accepted for presentation to the ApprovalPanel. 





SECTION B 

Guidance: This document is expected to be approximately 3 pages in length, but it can exceed this page limit. It is also permissible to carry forward content from this proposal to your later documents (e.g. functional specification) as appropriate. 

Your proposal must include at least the following sections. 

Introduction 

The goal of our fourth-year project is to develop an application that will allow users to input a keyword and using that keyword create a related dataset, scraped from a website such as Kaggle. The user can then input any desired values which are columns in the scraped dataset, and our algorithm will give an accurate prediction of the columns left empty. 

For example, a user wants an accurate prediction of the price of their home. The user inputs their keyword, something like “house\_price Dublin”. After that, the user can then input any of the characteristics of their own home that are columns in the scraped dataset, and our machine-learning algorithm will predict the empty inputs and give an accurate property price prediction.

Outline 

We plan on using python libraries such as BeautifulSoup4 and Selenium to successfully scrape datasets from one or more sources. The scraped dataset is then fed to our python code in which we will analyze the data in depth and identify the key markers/variables.

We will use the following generic python libraries to develop our predictive model. They include; NumPy, Pandas, Scikit-Learn and Matplotlib. We plan on getting the predictive model working using the functions in those libraries and once it is working as expected we will then optimise those functions, making them more efficient for our specific purpose.

We will then create a clean and straightforward web application using the Python Framework Django, where a user interfaces in which previously known values are inputted and predicted values are fed back to the interface. 

Background

Initially, the idea came from our CA4010 Data Warehousing and Mining project, where we are currently building a predictive model which can accurately predict the value of a house in Ireland. 

For our fourth-year project, we decided we could expand on this idea and create something that isn’t solely focused on house prices but could work for many things and also incorporated machine learning. We found that this project intrigued both of us and could be potentially very useful for others. 

We plan on using the information and skills gathered from CA4010 in our project which we believe will successfully help us in scraping, storing and analysing data for use. 

Achievements 

Ideally the function of this project would be for any user to input their desired category with a keyword and accurately predict a variable in that dataset. We want the project to have a user-orientated front page and at the same time have the bulk of our work behind the scenes where we scrape data and work with it. 

Justification

We both believe that this project could be useful in many different ways. Firstly in Ireland today, we are all aware of the crisis regarding housing. Both of us felt that we could develop a service that gave people the opportunity to get an accurate prediction of the price of their home years into the future. We believe that it would be particularly useful today for not only the majority of the people in Ireland thinking of selling their property but also for people looking to purchase their own property.

However, after discussing our proposed idea with our supervisor Mark Roantree, we came to the conclusion that our idea was too specific. We both felt that we could expand the idea making it useful in a number of different ways. For example, a user could predict the depreciation of the price of their car in a couple of years, or predict their risk of heart disease given the right variables. 

We hope for this project to be versatile and cover as many areas as the user intends. 

Programming language(s) 

- Python3
- Javascript 
- Jupyter notebook

Programming tools / Tech stack 

- We plan on using PyCharm and VSCode to facilitate the bulk of our code during this project. 
- Both of these offer many useful features and allow for inbuilt testing of code as well as creating a local server. 
- We will use MatPlotLib, Pandas, NumPy and scikit-learn to build our predictive model.
- We will be using DCU Computing Gitlab for our project repository.

Hardware 

No non-standard hardware planned so far. 

Learning Challenges 

- Building and testing the accuracy of our Predictive Model.
- Web scraping
- New Languages and libraries within Python, such as Jupyter notebook and Pandas within Python. 
- Returning the output from the source-code inside of our web application.

The breakdown of work 

Clearly identifies who will undertake which parts of the project. 

It must be clear from the explanation of this breakdown of work both that each student is responsible for separate, clearly defined tasks, and that those responsibilities substantially cover all of the work required for the project. 

Student 1: Kevin Boyle 

- Testing - Unit and Integration Testing Pipeline.
- Web Scraping - Selenium + BS4
- Manipulation of Datasets  - Organise and clean Datasets into a single one using Pandas

Student 2: Mark Reid

- Web-Application GUI - Javascript/CSS.
- Building the predictive model - MatPlotLib, Pandas, NumPy and scikit-learn.
- Testing the accuracy of our predictive model.

Over the course of the project, we will be using [Trello](https://trello.com) to report, assign and review our work and progress. 

We will also be assigning each other our Git merge requests.

![](Aspose.Words.0efea4c9-53de-4e1e-b212-4f591506b446.001.png)


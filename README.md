# Data-Analytics-project-01-using-Elasticsearch
<h1>Data Analytics based Optimization Project using Elasticsearch and GurobiPy </h1>

<h2>Description</h2>
This project is based on a case inspired from real-life, the dataset and the problem situation were provided by BELSIMPEL (A telecom company in the Netherlands). The product assortment of Belsimpel is rapidly expanding. As a result, their warehouse has become a bottleneck for their operations. Belsimpel does not intend to relocate the warehouse in the short-term. Therefore, it appears that the only way forward is to store a limited collection of products at the current warehouse and move the rest to a rental warehouse elsewhere. The advantage of the current warehouse is its efficiency. Thanks to the well-established organization of warehouse operations developed over the years and the availability of personnel with the right skills and experience, Belsimpel is able to offer nextday deliveries to its customers for any product that is stored at their current warehouse. It might not be possible to provide the  same  service quality for those products stored at a rental  warehouse—where deliveries may take several days, and, it is expected that this will have a significant impact on the sales. 
Belsimpel’s problem is to decide which products to keep in the current warehouse so that profit losses are kept at minimum. This allocation problem has ties with potential changes in the customer behavior in 
response  to  delivery  times  as  well  as the  organization  of  the operational  processes  of  managing inventories at the warehouse. (Problem description file and code file available in the folder)
<br />


<h2>Languages Used</h2>

- <b>Python 3 </b> 

<h2>Tools and Libraries Used</h2>

- <b>Elasticsearch and JSON for manipulating and querying big datasets </b>
- <b>Pandas for Dataframe operations and Numeric calculations </b>
- <b>NumPy for Numeric calculations </b>
- <b>Matplotlib & Seaborn for visualization (plotting charts) </b>
- <b>GurobiPy for optimizaion </b>

<h2>Program walk-through:</h2>

<p align="center">
Importing all necessary Libraries and ingesting the big dataset into Elasticsearch: <br/>
<img src="https://i.imgur.com/dUvnCpk.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Importing data after aggreagtion via json and storing final transformed data into a final dataframe:  <br/>
<img src="https://i.imgur.com/7fFMmFA.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<img src="https://i.imgur.com/HobLI7u.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Categorizing products based on profit margins and plotting histograms: <br/>
<img src="https://i.imgur.com/8D5gm2Q.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Calculation of base-stock & number of boxes required and plotting correlation plot:  <br/>
<img src="https://i.imgur.com/QcJlumo.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Identifying unique product couples based on correlation:  <br/>
<img src="https://i.imgur.com/CcmLu2v.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Generating the final product list and allocating the products based on first method:  <br/>
<img src="https://i.imgur.com/bVXsmrF.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Generating the final product list again and allocating the products based on second method:  <br/>
<img src="https://i.imgur.com/Kg73BUZ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
 Allocating the products based on third method (Optimization - Knapsack problem):  <br/>
<img src="https://i.imgur.com/27wYPJi.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />

</p>

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>

SDSS: Spectral Analysis

objective: explore how information from galactic spectra can be extracted using machine learning and how it can determine important physical quantities

Spectroscopic measurements can provide us with an unmatched amount of valuable information about astronomical objects. The abundance of such data in the Sloan Digital Sky Survey (SDSS) makes it one of the main astronomical sources to compare various machine learning models and their performance in case of spectral analysis.

Download optical spectra of galaxies and their corresponding physical parameters from the SDSS SkyServer! Select just a few physical parameters, e.g. redshift and magnitude, and try to determine them from the spectra! Treat spectra as “rows” in a tabular data set! First explore the data, then compare different regression methods (e.g. Linear Regression, Random Forest Regressor, Support Vector Regressor and Fully-connected Neural Networks) and try to find the best model! Provide an analysis of the results (e.g. AUC-ROC curves, confusion matrices, etc.).

Apply dimensionality reduction, e.g. PCA on the stellar spectra, choose an appropriate number of PCs to keep and retrain your models! How efficient is your model now?

What did you learn? Discuss the results and takeaways in your own words in a couple sentences!

Hints

SDSS data is publicly available and can be accessed through the SDSS SkyServer. There, it is stored in a relational database, which can be easily queried using the Structured Query Language (SQL). For convenience, the SkyServer ecosystem provides the CasJobs interface, which allows for the direct execution of SQL queries on the data servers and the download of the results in various formats. A modern alternative to CasJobs is called Betelgeuse, hosted on SciServer, a general purpose data science platform and the direct continuation of the SDSS SkyServer project. Beware! The total data size of SDSS is 652 TB! Download only a manageable size subset of it!
Photometric and spectroscopic data can be found in the `PhotoObjAll` and `SpecObjAll` tables, respectively. The two tables can be joined on the `specObjID` column, which is a unique identifier for each object with spectroscopic data in the SDSS. This column is present in both tables, so the join can be performed using the following SQL query:
```sql
		SELECT <desired columns>
	FROM PhotoObjAll AS p
	JOIN SpecObjAll AS s
	ON p.specObjID = s.specObjID
	```
Use the `pandas` library to read data into a `DataFrame` and the `scikit-learn` package to build the ML models and use `pytorch` (or `keras`) to build neural networks
Study the excellent documentations of `pandas`, `scikit-learn` or other libraries if you want to know more about specific functions

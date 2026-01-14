Report – Historical Flight Delay and Weather Data USA

In my project, I performed analysis and predictive modeling based on the dataset “Historical Flight Delay and Weather Data USA (May–December 2019)”, obtained from the Kaggle platform.
The dataset contains flight information and related weather data collected from meteorological stations.
After merging eight monthly CSV files, I obtained over 5.5 million records and 35 attributes.
The dataset includes flight identifiers, operational dates and times, delay information, and weather conditions.
The goal of my project was to build a classifier that predicts whether a given flight will be delayed.
First, I cleaned the data. I found a significant number of missing values in the columns actual_arrival_dt, actual_departure_dt, and tail_number, which I removed because they were not useful for predicting future delays.
For missing weather values, I filled them with the median. Next, I performed categorical encoding (carrier_code, origin_airport, destination_airport, cancelled_code) using One-Hot Encoding, which increased the number of columns to 785.
To reduce RAM usage, I optimized numerical types (float64 → float32 and int64 → int32). Since the full dataset was too large to train locally, I used downsampling and created test samples of 5,000, 50,000, and 75,000 observations.
I ultimately kept 75,000, ensuring a good balance between training time and model quality.
I normalized features using StandardScaler, then split the data into training (80%) and test (20%) sets. 
I tested eight required machine learning models: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting, and MLP.
I evaluated the models based on accuracy, precision, recall, F1-score, and AUC, also using probability predictions whenever possible.
The best results were achieved with Gradient Boosting, which reached approximately 0.995 accuracy, along with very high F1-score (≈0.986) and AUC (≈0.997). 
The results were stable across both smaller samples (5000) and the larger 75,000 sample.
Other models such as Logistic Regression, MLP, and Decision Tree also performed very well, with F1-scores in the 0.96–0.98 range.
The weakest results came from SVM, KNN, and Naive Bayes, which struggled with the high dimensionality of the data after One-Hot Encoding.
For the best model, I performed hyperparameter tuning using RandomizedSearchCV, which confirmed the strong stability of Gradient Boosting.
I then generated a confusion matrix and ROC curve, which visually confirmed good class separation and very low false prediction rates.
The dataset contained columns strongly correlated with the target variable, such as departure_delay, delay_weather, and delay_carrier, which significantly helped the model classify the data.
This means the problem was relatively easy because the model relied on features containing direct signals of delay. 
In this project, I did not predict future delays; I only classified delays based on historical data, which is a limitation.
Additionally, the dataset covers only one year, which may not account for seasonal or exceptional events.

￼
 <img width="634" height="475" alt="Confusion Matrix- GradientBoosting (tuned)" src="https://github.com/user-attachments/assets/c8a61c13-064c-4a6c-975e-5e599ce3e308" />

 <img width="634" height="478" alt="Positive" src="https://github.com/user-attachments/assets/b79733eb-886b-46a0-9aa8-c73027cda14f" />


￼

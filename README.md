Report – Historical Flight Delay and Weather Data USA

In this project, I performed an end-to-end analysis and developed predictive machine learning models using the dataset “Historical Flight Delay and Weather Data USA (May–December 2019)” obtained from the Kaggle platform. The dataset contains detailed flight information along with corresponding weather data collected from meteorological stations across the United States. After importing and merging eight monthly CSV files, I obtained a combined dataset of over 5.5 million records and 35 attributes, including identifiers, timestamps, delay details, and weather measurements. The primary goal of the project was to build a classifier capable of predicting whether a given flight would be delayed.

 The initial dataset contained several columns with substantial missing values, specifically actual_arrival_dt, actual_departure_dt, and tail_number. Since these features were not useful for predicting future delays—and because they contained too many missing values to be reliably imputed—they were removed. Missing values in weather-related columns were filled using median imputation to maintain consistency and preserve the underlying structure of the data. This ensured that the dataset remained coherent without introducing statistical bias.

 I then transformed categorical variables, such as carrier_code, origin_airport, destination_airport, and cancelled_code, using One-Hot Encoding. This significantly increased the dimensionality of the dataset from 35 to 785 columns. To optimize memory usage and speed up processing time, numerical data types were converted from float64 to float32 and from int64 to int32. These adjustments allowed the models to run efficiently on a local machine without exhausting system resources.

 Because the full dataset was too large to process in its entirety, I performed downsampling and prepared three subsets containing 5,000, 50,000, and 75,000 rows. After evaluating the computational load and performance, I selected the 75,000-row subset, which provided a strong balance between training time and predictive quality.

 The numerical features were scaled using StandardScaler, and the dataset was split into 80% training data and 20% test data. I trained eight machine learning models, including Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting, and MLP. Each model was evaluated using accuracy, precision, recall, F1-score, and AUC, ensuring a comprehensive comparison of their predictive abilities.

 Among all models, Gradient Boosting achieved the highest performance, reaching an accuracy of approximately 0.995, an F1-score of about 0.986, and an AUC close to 0.997. These results remained consistent across both smaller and larger downsampled datasets, indicating strong stability and robustness. Logistic Regression, MLP, and Decision Tree also produced strong results, with F1-scores ranging from 0.96 to 0.98. In contrast, SVM, KNN, and Naive Bayes performed poorly due to the high-dimensional feature space created by One-Hot Encoding.

 To refine the best-performing model, I conducted hyperparameter tuning using RandomizedSearchCV. This tuning phase confirmed the reliability of the Gradient Boosting model without significantly altering its already excellent performance. Visual evaluations using a confusion matrix and ROC curve revealed clear separation between delayed and non-delayed classes and demonstrated minimal false predictions.

 During the analysis, I identified several features—such as departure_delay, delay_weather, and delay_carrier—that were highly correlated with the target variable. These variables directly reflected delay conditions, making the classification task relatively simple. As a result, the model primarily identified delays based on explicit delay information rather than predicting future delays. This indicates a limitation in the real-world applicability of the model if used for proactive forecasting.

 Another limitation arises from the temporal scope of the dataset, which includes only eight months of data. Flight delays are influenced by seasonal patterns, operational cycles, and rare events, none of which can be fully captured with such a short dataset. Despite these constraints, the project successfully demonstrates the complete machine learning workflow, including preprocessing, feature engineering, model training, hyperparameter tuning, and performance evaluation.

 In the future, the analysis could be extended by restricting features to those available before departure, collecting multi-year datasets, experimenting with time-series forecasting techniques, or deploying the model for real-time predictions.

￼
 <img width="634" height="475" alt="Confusion Matrix- GradientBoosting (tuned)" src="https://github.com/user-attachments/assets/c8a61c13-064c-4a6c-975e-5e599ce3e308" />

 <img width="634" height="478" alt="Positive" src="https://github.com/user-attachments/assets/b79733eb-886b-46a0-9aa8-c73027cda14f" />


￼

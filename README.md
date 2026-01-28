Predicting Baseball Salaries Using Big Data
Tools
Shira, Julia, Abby, and Jason
Introduction
We are interested in the predictive power of MLB statistics. The MLB takes meticulous statistics
that measure players' performances per season; we wanted to use these numbers to predict player salaries
in future seasons. In our analysis, we will explore the relationship between hitting statistics and salary.
This analysis offers practical value to MLB franchises in providing a model for how much a player of any
given performance might or should earn. It also provides insight into which factors contribute most
significantly to determining salary level.
As mentioned, the variable we are predicting is salary (data from USA Today), and we are using
numerous hitting statistics provided by the MLB to predict it. We have chosen hitting statistics over
pitching statistics as every baseball player hits while only a select few pitch. Thus, we cover a
comprehensive range of baseball players in our data analysis.
We are training on data on statistics from the 2020-2024 seasons, and testing on statistics from the
2025 season. We will use the linear regression, KNN regression, and the random forest algorithm to
analyze the relationship between hitting statistics and salary. Linear regression provides a basic model for
analyzing relationships, while KNN regression offers a predictive model based on k-nearest neighbors.
Random forest furthers our examination by averaging out predictions from many different models in
order to improve our prediction accuracy.
Furthermore, we will conduct exploratory data analysis with K-Means analysis. K-Means analysis
will provide some insight into which factors contribute most significantly to determining salary level.
Data Acquisition
We got our data from two separate places. First, we used the official MLB website’s API to fetch
hitter statistics from 2020 through 2025. The statistics available from MLB are:
['age'
,
'gamesPlayed'
,
'groundOuts'
'airOuts'
'runs'
'doubles'
,
,
,
,
'triples'
'homeRuns'
,
,
'strikeOuts'
,
'baseOnBalls'
'intentionalWalks'
'hits'
,
,
,
'hitByPitch'
,
'avg'
'atBats'
,
,
'obp'
,
'slg'
,
'ops'
,
'caughtStealing'
'stolenBases'
,
,
'stolenBasePercentage'
,
'caughtStealingPercentage'
,
'groundIntoDoublePlay'
'numberOfPitches'
,
,
'plateAppearances'
'totalBases'
'rbi'
'leftOnBase'
,
,
,
,
'sacBunts'
'sacFlies'
'babip'
'groundOutsToAirouts'
'catchersInterference'
'atBatsPerHomeRun'
,
'player
,
name'
,
_
,
,
,
,
'player
_
id']
Second, we fetched MLB opening day salaries from a Google spreadsheet made available on
Cot's Baseball Contracts. We merged the hitting stats and salary dataframes along the player name and
season columns to get our final dataset.
1
Data Cleaning
Before jumping into the analysis stage, the data
needed some slight cleaning. First, we performed a PCA to
reduce the high dimensional data points to easily-visualized
two dimensions. This revealed that data from 2020 was very
abnormal compared to every other season—which we
extrapolate to be a result of the COVID-19 pandemic.
Next, since there is a small minority of star-player
salaries which are much larger than the average player salary,
we performed a log transform on the salaries. This will create
a more linear relationship between salary and the other
variables. Finally, we split the data into a training set (2021
through 2024) and a testing set (2025) to see if we’d be able
to use historical trends to predict future salaries.
Linear Regression Analysis
We began with linear regression to determine the predictive power of players’ statistics on their salaries.
Variable Selection
First, we used variable selection to reduce the number of variables in our analysis from the 36 variables in
our dataset to the 11 most important variables. We removed low variance variables and variables with
high correlations or collinearity, prioritizing for well-known metrics.
In the end, our 11 variables for the linear regression analysis are:
Player, Season, Age, RBI, Average, Slugging, Home Runs, On Base Percentage, Triples, Intentional
Walks, and Stolen Bases.
2
Controlling for Within-Player Correlations
Since our dataset extends over multiple years and players appear multiple times across the years, we have
to control for these repeat occurrences. There are two main ways to address this: a fixed effects model or a
mixed-effects model. We opted for the mixed-effects model, which creates random intercepts as baseline
salaries for each player, because it is simpler and handles the correlations better.
Transformations
We assessed if any variables required transformations. The salary output required a log transformation
due to the large values. The Aikake Information Criterion score validated this decision as it improved
from 16909 to 1271.
To explore any other transformations, we visualized each independent variable’s residual plot and also
printed each skew metric. Three variables had significant skew metrics; however, no transformations
improved the AIC score, so none were implemented.
Final Model:
log(salaryit
+1) = β0
+ β1 (avgit ) + β2( rbiit ) + β3( homeRunsit ) + β4( slgit ) + β5( obpit ) + β6 (triplesit ) +
β7( stolenBasesit) + β8( intentionalWalksit) + β9 (ageit ) + β10( seasonit ) + ui
+ εit
ui
∼ N(0,σu
2
) is the random intercept for
player i
εit ∼ N(0,σ2) is the residual error
β0,β1,…,β10β0 ,β1 ,…,β10 are
fixed-effect coefficients
The R-squared value was 0.885, and the
Residual Mean Square Error was 0.46.
These indicate that our model does a
reasonably good job of modeling the
relationship between our independent
and dependent variables.
Home Runs, Intentional Walks, Age,
Season, and Player were the variables
statistically significant at the ɑ = 0.05
level.
This can be interpreted as such:
❖
❖
❖
❖
A player with one home run more than another is predicted to be paid 3.98% more.
A player with one intentional walk more than another is predicted to be paid 3.77% more.
A player one year older than another is predicted to be paid 29% more.
From one season to another, a player is predicted to be paid 16.6% more.
3
Test on 2025 Data:
We tested this model on 2025 data. Below is our predicted versus actual plot.
These two plots demonstrate that our model does a relatively good job at predicting salaries for baseball
players. The biggest weakness is predictions for low-salaried players, as the log scale significantly
compresses these outcomes into the vertical clump seen in the bottom left hand corner on both graphs.
This is also a result of the compressed random intercepts and predictor variance.
Here are our error results:
Metric Train Test
R-Squared 0.885 0.521
MSE 0.211 0.937
RMSE 0.460 0.968
We see that the model does quite a good job at predicting salaries on the training data, but the accuracy
drops significantly and the error increases when it comes to the test data. Thus, we tried other methods of
analysis.
KNN Analysis:
We next applied a K-Nearest Neighbor (KNN) model to determine whether a distance-based
method could improve our ability to predict player salaries from performance statistics. Unlike linear
4
regression, KNN does not assume linear relationships; instead, it learns patterns by comparing each player
to their statistically “closest” neighbors.
Variable Selection:
Because KNN is strongly affected by the predictor features, we performed targeted variable
selection before modeling. Starting from the full set of 36 variables, we selected 5 core performance
metrics that are widely recognized as strong indicators of offensive production. The final features selected
for the KNN model were Age, On-base plus slugging (OPS), Home Runs, Slugging Percentage(SLG),
and Base on Balls (Walks)
These features were chosen because they showed meaningful correlations with salary, the most
important key components of offensive, or hitting, evaluations. And they reduce the dimensionality. The
higher the dimensionality, the more negatively affected the KNN models are since they are
distance-based.
Again, we chose to train on all seasons except 2020 to avoid contamination of data as described
earlier, and also used the Log transformation of Salary. The Player salaries are highly right-skewed, sings
a small number of elite players are earning extremely high salaries. To address these issues, we performed
a log transformation to stabilize the variance. To convert the salary prediction into classification, we used
bins to log the salaries into 3 categories: Low, medium, and high salaries. This was created using the
global minimum and maximum log-salaries to avoid generating NaN categories. This ensures that all
players in both training and testing fall into one of these groups. We trained a KNN classifier with k = 20
nearest neighbors. The model predicts the salary category of each 2025 player by examining which salary
group their nearest statistical neighbor belongs to.
Assessment:
To estimate the individual salaries more precisely, we applied the KNN regression model using
the log of the player salaries. The Model resulted in:
The R2 value indicates that the model explains just over
half of the variation in log-salary. This represents moderate
predictability. The RMSE of 0.9194 indicates a meaningful
prediction error, which corresponds to salary deviations,
which makes sense given the variability of MLB salaries.
A 72.1% accuracy rate represents a moderate predictive
power, especially given the simplicity of the KNN
algorithm. This indicates that salary grouping remains
difficult due to nonlinearities and other contextual factors such as contracts, service time, etc.
Metric Test
R-Squared 0.5680
MSE 0.8453
RMSE 0.9194
5
K-Means Analysis:
We first conducted a K-Means clustering of all numeric variables of MLB statistics. We used
StandardScaler to give each variable an equal impact on variance. We arbitrarily chose n=6 for clusters.
As seen in Figure 3.1, the data formed in vertical lines, likely because of the variable age which has
discrete categories.
Then, we constructed an ‘elbow’ graph (see Figure 3.2) which, in addition with the inertia list
(see Figure 3.3), displays that the rate of inertia significantly slows down at k =2. The optimal number of
clusters is likely 2 (see Figure 3.4) which balances cluster compactness and separation as well as
minimizing with-cluster variance.
6
Now that there are two distinct clusters of MLB players, we wanted to discern their differences.
We calculated the average of each variable for each cluster (see Figure 3.6).
FIGURE 3.6
As the output displays, cluster 1 seems to perform better across most variables, including
homeRuns (23.99 v 17.86), rbi (76.35 v. 63.82), age (30.71 v. 27.23), gamesPlayed (133.28 v.130.59), and
log_
salary (16.89 v. 14.66). This insinuates that cluster 1 grouped experienced and more talented players,
while cluster 0 contains perhaps younger,
‘less’ talented players. Thus, the elbow method and this chart
suggest that there are two distinct groups of players, statistically discernible by differences in experience
and talent.
Next, considering the large number of variables in this dataset, we used PCA to form clusters
using the most significant variables. We set PCA to account for 90% of the variance in the dataset, it
returned that 13 components of all the numeric variables in MLB statistics accounted for the 90%. We
then created a visual of the K-Means clustering with n=2 clusters (elbow method) using 13 components
(PCA).
Below is a side-by-side comparison of n=2 clusters before (left) and after (right) of PCA reduction.
FIGURE 3.5
Last, we created a chart (see Figure 3.7) that displays the 3 most impactful variables factoring into each
PCA. The chart shows that there is a wide variety of variables that factor into forming the clusters created
by K-Means with n=2 clusters (per elbow method) based on 13 components (per PCA). For example,
7
homeRuns, a flashy statistic, is only in the top 3 variables for 1 PCA (PCA 12). This signals that many
variables contribute to defining the patterns in this dataset, and that variables like homeRuns aren’t as
significant as one might think.
Random Forest Regression Analysis:
The random forest strategy in machine learning is an ensemble method that combines many
decision trees for both regression and classification tasks. While a single decision tree is at high risk of
overfitting, a random forest reduces this risk by training multiple trees on different random samples of the
data, with each tree also considering only a random subset of features at each split. The final prediction is
made by averaging (regression) or voting (classification) across all trees.
Random forest regression was reasonably successful at predicting 2025 salaries after training on
the 2021 through 2024 data.
Metric Value
R2 0.5377640595446833
MSE 0.9043834797254332
RMSE 0.950990788454564
One benefit of random forest
methods is that it is very easy to evaluate
feature importance. A popular metric for determining feature importance is the Gini Index, which
measures how often a randomly chosen element would be incorrectly classified when a feature is
excluded from the decision trees. We were curious to see which features were most important for
determining baseball salary. Interestingly, age is by far the most important feature, accounting for
approximately 80% of the decision. I was curious to see the correlations between each of the most
impactful variables on their own against salary.
8
Next Steps: Incorporating Popularity Metric and Pitching Data
Although these various methods were able to estimate salary with moderate success, there is still
a lot of room for improvement. In particular, our data is likely lacking all of the necessary information
that goes into determining salary. We see two large pitfalls in the data which could hopefully improve our
model performance in the future.
First, we only used MLB data for hitting, so we’re missing all of the pitching statistics. This
likely biases our estimates to undervalue great pitchers.
Second,
baseball player
popularity brings in a lot
of money for teams, so we
suspect this plays a big
role in salary as well. We
came up with a number of
proxies to measure
popularity, but
unfortunately faced API
paywalls which
prevented us from
incorporating them into our
dataset. The places we
thought of extracting
popularity data from
were: Google Search
Trends Data, YouGov
Public Opinion
Data, and Instagram
Follower Data.
Conclusions:
Across all the different models and analyses, our central goal was to determine how effectively
MLB hitting statistics can predict player salaries. We also inquired as to which performance metrics
matter most. Using the four approaches, linear regression, KNN, random forest, and K-means clustering,
we noticed several consistent themes.
All predictive models achieved moderate accuracy, with R2 scores generally between 0.50 and .60
on the 2025 testing data. This shows that hitting metrics contain somewhat meaningful information about
salary, but they cannot fully explain salary variation. This was semi-expected because MLB salaries are
9
heavily impacted by the non-performance factors such as contract structure, attribution years, market size,
player popularity, etc.
Across linear regression, KNN, and especially random forest, age was the single strongest
predictor of salary. This reflects real-world salary structures. Players gain negotiating power as they
accrue service time and approach free agency. Players cannot negotiate freely when they are young; they
first have to move through the salary phases. The first phase is pre-arbitration, where for around 3 years,
players are stuck earning the league minimum salary. Then there is the salary arbitration, where players
earn raises based on comparison to similar players, which lasts for another 3 years. Then the players reach
the free agency term, where they are able to negotiate huge contracts on the open market. So once players
hit free agency, we tend to see salaries spike dramatically since the older players are in the last 20 and
early 30s dominating the top salary bucket.
Linear regression identified home runs, intentional walks, and slugging metrics as statistically
significant.
KNN found that players with similar OPS, HR, SLG, and BB tend to end up in similar salary categories
Random forest found that age is ranked far above specific batting metrics
This all indicates that while offensive production matters, MLB teams do not reward hitters solely based
on one or two “flashy” statistics; many metrics contribute a small amount each.
Looking strictly at the K-means analysis, the older, more experienced, higher performing, and
higher paid players lay in cluster 1, while cluster 0 had younger, lower experience, and lower paid
players. This once again reinforces that age is a major salary driver.
Model R² (Test) MSE (Test) RMSE (Test)
Linear Regression 0.521 0.937 0.968
KNN Regression 0.568 0.845 0.919
Random Forest
Regression
0.538 0.904 0.951
KNN had the highest R² and lowest error, but only slightly better than random forest. Random
forest performed comparably but did not outperform simpler methods. Linear regression fit the training
data very well (R² = 0.885) but dropped substantially on the test set, suggesting mild overfitting or
limitations of linearity. So overall, no model achieved high or perfect predictive accuracy.
MLB hitting stats alone can explain roughly half of the variation in player salary. The most
reliable predictor is age, but no single performance statistic determines salary. Future improvements, such
10
as incorporating pitching data, popularity metrics, contract status, or other contextual variables, would
likely produce significantly more accurate models.
11

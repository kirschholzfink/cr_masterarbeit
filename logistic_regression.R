# install.packages("RMariaDB")
# install.packages("dplyr")
# install.packages("ggpubr")
# install.packages("ggplot2")

library(DBI)
library("ggplot2")
library("dplyr")
library("ggpubr")
library("car")

# Connect to MySQL database
con <- dbConnect(
  RMariaDB::MariaDB(),
  dbname = "db_openstack",
  host = "localhost",
  port = 3306,
  user = "root",
  password = "password"
)

# Save db query result to df variable.
df <-
  dbGetQuery(
    con,
    'select ch_patchSetCount, ch_affectedFilesCount, ch_churnSize, ch_initialResponseTimeInHours, ch_authorSentimentAsAvg, ch_reviewerSentimentAsAvg,
    ch_authorSentiment, ch_reviewerSentiment
 from t_change'
  )

dbDisconnect(con)

str(df)

# -------------------
# INSPECT DISTRIBUTION OF NEGATIVE AND NON-NEGATIVE SENTIMENTS FOR AUTHORS AND REVIEWERS

# Display numeric representation of negative and non-negative sentiment class in author sentiments
df.table <- table(df$ch_authorSentiment)
df.table

# Display numeric representation of negative and non-negative sentiment class in reviewer sentiments
df.table <- table(df$ch_reviewerSentiment)
df.table

# Display numeric representation of negative and non-negative sentiment class in averaged author sentiments
df.table <- table(df$ch_authorSentimentAsAvg)
df.table

# Display numeric representation of negative and non-negative sentiment class in averaged reviewer sentiments
df.table <- table(df$ch_reviewerSentimentAsAvg)
df.table


# -------------------
# ADD ADDITIONAL COLUMNS TO REASSIGN CATEGORICAL AND BINARY SENTIMENT VALUES FOR REGRESSION ANALYSIS AND GRAPHS

# Add column containing binary values 1 and 0 for String values 'negative' and 'non-negative' in ch_authorSentiment.
df$ch_authorSentiment.bin <-
  ifelse(
    test = df$ch_authorSentiment == 'negative',
    yes = 1,
    no = 0
  )

# Add column containing String values 'negative' and 'non-negative' for binary values 1 and 0 in ch_authorSentimentAsAvg.
df$ch_authorSentimentAsAvg.cat <-
  ifelse(
    test = df$ch_authorSentimentAsAvg == 1,
    yes = 'negative',
    no = 'non-negative'
  )

# Add column containing binary values 1 and 0 for String values 'negative' and 'non-negative' in ch_reviewerSentiment.
df$ch_reviewerSentiment.bin <-
  ifelse(
    test = df$ch_reviewerSentiment == 'negative',
    yes = 1,
    no = 0
  )

# Add column containing String values 'negative' and 'non-negative' for binary values 1 and 0 in ch_reviewerSentimentAsAvg.
df$ch_reviewerSentimentAsAvg.cat <-
  ifelse(
    test = df$ch_reviewerSentimentAsAvg == 1,
    yes = 'negative',
    no = 'non-negative'
  )

str(df)

# -------------------
# EXPLORING THE DATASET

# Author sentiment: Initial Response Time
ggscatter(
  df,
  x = 'ch_initialResponseTimeInHours',
  y = 'ch_authorSentimentAsAvg.cat',
  xlab = 'Initial Response Time (in hours)',
  ylab = 'Author Sentiment',
  
  color = 'ch_authorSentimentAsAvg.cat',
  
  palette = c("#e86530", "#f28c02"),
  
  mean.point = TRUE,
  
  rug = TRUE,
  font.label = 10,
  repel = TRUE
)

# Author sentiment: Churn Size
ggscatter(
  df,
  x = 'ch_churnSize',
  y = 'ch_authorSentimentAsAvg.cat',
  xlab = 'ChurnSize',
  ylab = 'Author Sentiment',
  
  color = 'ch_authorSentimentAsAvg.cat',
  
  palette = c("#e86530", "#f28c02"),
  
  mean.point = TRUE,
  
  rug = TRUE,
  font.label = 10,
  repel = TRUE
)

# Author sentiment: Patch Set Count
ggscatter(
  df,
  x = 'ch_patchSetCount',
  y = 'ch_authorSentimentAsAvg.cat',
  xlab = 'Patch Set Count',
  ylab = 'Author Sentiment',
  
  color = 'ch_authorSentimentAsAvg.cat',
  
  palette = c("#e86530", "#f28c02"),
  
  mean.point = TRUE,
  
  rug = TRUE,
  font.label = 10,
  repel = TRUE
)

# Reviewer sentiment: Initial Response Time
ggscatter(
  df,
  x = 'ch_initialResponseTimeInHours',
  y = 'ch_reviewerSentimentAsAvg.cat',
  xlab = 'Initial Response Time (in hours)',
  ylab = 'Reviewer Sentiment',
  
  color = 'ch_reviewerSentimentAsAvg.cat',
  
  palette = c("#e86530", "#f28c02"),
  
  mean.point = TRUE,
  
  rug = TRUE,
  font.label = 10,
  repel = TRUE
)

# Reviewer sentiment: Churn Size
ggscatter(
  df,
  x = 'ch_churnSize',
  y = 'ch_reviewerSentimentAsAvg.cat',
  xlab = 'Churn Size',
  ylab = 'Reviewer Sentiment',
  
  color = 'ch_reviewerSentimentAsAvg.cat',
  
  palette = c("#e86530", "#f28c02"),
  
  mean.point = TRUE,
  
  rug = TRUE,
  font.label = 10,
  repel = TRUE
)

# Reviewer sentiment: Patch Set Count
ggscatter(
  df,
  x = 'ch_patchSetCount',
  y = 'ch_reviewerSentimentAsAvg.cat',
  xlab = 'Patch Set Count',
  ylab = 'Reviewer Sentiment',
  
  color = 'ch_reviewerSentimentAsAvg.cat',
  
  palette = c("#e86530", "#f28c02"),
  
  mean.point = TRUE,
  
  rug = TRUE,
  font.label = 10,
  repel = TRUE
)

# -------------------
# PERFORM LOGISTIC REGRESSION ANALYSIS

# Check independent variables for multicollinearity by means of Variance Inflation Factor (VIF) method.

vif.model.auth <-
  glm(
    factor(ch_authorSentimentAsAvg) ~ ch_initialResponseTimeInHours + ch_patchSetCount + ch_affectedFilesCount + ch_churnSize,
    data = df,
    family = 'binomial'
  )

vif(vif.model.auth)

vif.model.rev <-
  glm(
    factor(ch_reviewerSentimentAsAvg) ~ ch_initialResponseTimeInHours + ch_patchSetCount + ch_affectedFilesCount + ch_churnSize,
    data = df,
    family = 'binomial'
  )

vif(vif.model.rev)

# ch_affectedFilesCount and ch_churnSize are highly correlated as predictors of author sentiment, which is why ch_affectedFilesCount is omitted as a factor in the model below:

# Regression analysis for author sentiment.

auth.model <-
  glm(
    ch_authorSentimentAsAvg ~ ch_initialResponseTimeInHours + ch_patchSetCount + ch_churnSize,
    data = df,
    family = 'binomial'
  ) 

summary(auth.model)

# Calculating the odds ratio (OR) for the author sentiment logistic classifier.

exp(coef(auth.model))

# Regression analysis for reviewer sentiment.

rev.model <-
  glm(
    ch_reviewerSentimentAsAvg ~ ch_initialResponseTimeInHours + ch_patchSetCount + ch_churnSize,
    data = df,
    family = 'binomial'
  ) 

summary(rev.model)

# Calculating the odds ratio (OR) for the reviewer sentiment logistic classifier.

exp(coef(rev.model))

saveRDS(df, 'output.rds')

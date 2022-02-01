# install.packages("RMariaDB")
# install.packages("dplyr")
# install.packages("ggpubr")
install.packages("ggplot2")

library(DBI)
library("dplyr")
library("ggpubr")

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
    'select id, ch_status, ch_patchSetCount, ch_affectedFilesCount, ch_churnSize, ch_initialResponseTimeInHours, ch_authorialSentiment
 from t_change'
  )

dbDisconnect(con)

str(df)

# Display numeric representation of negative and non-negative sentiment class
df.table <- table(df$ch_authorialSentiment)
df.table

# Reassign binary values 0 and 1 to String values 'negative' and 'non-negative' in ch_authorialSentiment column.
df$ch_authorialSentiment.new <-
  ifelse(
    test = df$ch_authorialSentiment == 'negative',
    yes = 1,
    no = 0
  )

# Perform logistic regression analysis.
log.model <-
  glm(
    ch_authorialSentiment.new ~ ch_initialResponseTimeInHours,
    data = df,
    family = 'binomial'
  )

# Display regression analysis results.
summary(log.model)

barplot(df$ch_initialResponseTime)

summary(log.model)$coefficients

str(summary(log.model))

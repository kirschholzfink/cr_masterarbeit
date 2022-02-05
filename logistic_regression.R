# install.packages("RMariaDB")
# install.packages("dplyr")
# install.packages("ggpubr")
# install.packages("ggplot2")

library(DBI)
library("ggplot2")
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
 from t_change where ch_initialResponseTimeInHours < 50'
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

str(df)
ggdensity(
  df,
  x = 'ch_patchSetCount',
  add = 'mean',
  color = 'ch_authorialSentiment'
  ,
  fill = 'ch_authorialSentiment',
  palette = c('#00AFBB', '#E7B800')
)

gghistogram(
  df,
  x = 'ch_initialResponseTimeInHours',
  add = 'mean',
  color = 'ch_authorialSentiment'
  ,
  fill = 'ch_authorialSentiment',
  bins=200,
  palette = c('#00AFBB', '#E7B800')
)

ggscatter(df, x = 'ch_patchSetCount', y = 'ch_authorialSentiment',
          
          color = 'ch_authorialSentiment',
          
          palette = c('#00AFBB', '#E7B800', '#FC4E07'),
                       
                       ellipse = TRUE, mean.point = TRUE,
                       
                       rug = TRUE, font.label = 10, repel = TRUE)

# Perform logistic regression analysis.
log.model <-
  glm(
    ch_authorialSentiment.new ~ ch_affectedFilesCount,
    data = df,
    family = 'binomial'
  )

# Display regression analysis results.
summary(log.model)

saveRDS(df, 'output.rds')

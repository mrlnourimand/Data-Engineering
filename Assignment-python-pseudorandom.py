# Databricks notebook source
# MAGIC %md
# MAGIC # Data-Intensive Programming - Group assignment
# MAGIC
# MAGIC This is the **Python** version of the assignment. Switch to the Scala version, if you want to do the assignment in Scala.
# MAGIC
# MAGIC In all tasks, add your solutions to the cells following the task instructions. You are free to add new cells if you want.<br>
# MAGIC The example outputs, and some additional hints are given in a separate notebook in the same folder as this one.
# MAGIC
# MAGIC Don't forget to **submit your solutions to Moodle** once your group is finished with the assignment.
# MAGIC
# MAGIC ## Basic tasks (compulsory)
# MAGIC
# MAGIC There are in total nine basic tasks that every group must implement in order to have an accepted assignment.
# MAGIC
# MAGIC The basic task 1 is a separate task, and it deals with video game sales data. The task asks you to do some basic aggregation operations with Spark data frames.
# MAGIC
# MAGIC The other basic coding tasks (basic tasks 2-8) are all related and deal with data from [https://figshare.com/collections/Soccer_match_event_dataset/4415000/5](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5) that contains information about events in [football](https://en.wikipedia.org/wiki/Association_football) matches in five European leagues during the season 2017-18. The tasks ask you to calculate the results of the matches based on the given data as well as do some further calculations. Special knowledge about football or the leagues is not required, and the task instructions should be sufficient in order to gain enough context for the tasks.
# MAGIC
# MAGIC Finally, the basic task 9 asks some information on your assignment working process.
# MAGIC
# MAGIC ## Advanced tasks (optional)
# MAGIC
# MAGIC There are in total of four advanced tasks that can be done to gain some course points. Despite the name, the advanced tasks may or may not be harder than the basic tasks.
# MAGIC
# MAGIC The advanced task 1 asks you to do all the basic tasks in an optimized way. It is possible that you gain some points from this without directly trying by just implementing the basic tasks efficiently. Logic errors and other issues that cause the basic tasks to give wrong results will be taken into account in the grading of the first advanced task. A maximum of 2 points will be given based on advanced task 1.
# MAGIC
# MAGIC The other three advanced tasks are separate tasks and their implementation does not affect the grade given for the advanced task 1.<br>
# MAGIC Only two of the three available tasks will be graded and each graded task can provide a maximum of 2 points to the total.<br>
# MAGIC If you attempt all three tasks, clearly mark which task you want to be used in the grading. Otherwise, the grader will randomly pick two of the tasks and ignore the third.
# MAGIC
# MAGIC Advanced task 2 continues with the football data and contains further questions that are done with the help of some additional data.<br>
# MAGIC Advanced task 3 deals with some image data and the questions are mostly related to the colors of the pixels in the images.<br>
# MAGIC Advanced task 4 asks you to do some classification related machine learning tasks with Spark.
# MAGIC
# MAGIC It is possible to gain partial points from the advanced tasks. I.e., if you have not completed the task fully but have implemented some part of the task, you might gain some appropriate portion of the points from the task. Logic errors, very inefficient solutions, and other issues will be taken into account in the task grading.
# MAGIC
# MAGIC ## Assignment grading
# MAGIC
# MAGIC Failing to do the basic tasks, means failing the assignment and thus also failing the course!<br>
# MAGIC "A close enough" solutions might be accepted => even if you fail to do some parts of the basic tasks, submit your work to Moodle.
# MAGIC
# MAGIC Accepted assignment submissions will be graded from 0 to 6 points.
# MAGIC
# MAGIC The maximum grade that can be achieved by doing only the basic tasks is 2/6 points (through advanced task 1).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Short summary
# MAGIC
# MAGIC ##### Minimum requirements (points: 0-2 out of maximum of 6):
# MAGIC
# MAGIC - All basic tasks implemented (at least in "a close enough" manner)
# MAGIC - Moodle submission for the group
# MAGIC
# MAGIC ##### For those aiming for higher points (0-6):
# MAGIC
# MAGIC - All basic tasks implemented
# MAGIC - Optimized solutions for the basic tasks (advanced task 1) (0-2 points)
# MAGIC - Two of the other three advanced tasks (2-4) implemented
# MAGIC     - Clearly marked which of the two tasks should be graded
# MAGIC     - Each graded advanced task will give 0-2 points
# MAGIC - Moodle submission for the group

# COMMAND ----------

# import statements for the entire notebook
# add anything that is required here

import re
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from functools import reduce

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 1 - Video game sales data
# MAGIC
# MAGIC The CSV file `assignment/sales/video_game_sales.csv` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains video game sales data (based on [https://www.kaggle.com/datasets/patkle/video-game-sales-data-from-vgchartzcom](https://www.kaggle.com/datasets/patkle/video-game-sales-data-from-vgchartzcom)).
# MAGIC
# MAGIC Load the data from the CSV file into a data frame. The column headers and the first few data lines should give sufficient information about the source dataset. The numbers in the sales columns are given in millions.
# MAGIC
# MAGIC Using the data, find answers to the following:
# MAGIC
# MAGIC - Which publisher has the highest total sales in video games in North America considering games released in years 2006-2015?
# MAGIC - How many titles in total for this publisher do not have sales data available for North America considering games released in years 2006-2015?
# MAGIC - Separating games released in different years and considering only this publisher and only games released in years 2006-2015, what are the total sales, in North America and globally, for each year?
# MAGIC     - I.e., what are the total sales (in North America and globally) for games released by this publisher in year 2006? And the same for year 2007? ...
# MAGIC

# COMMAND ----------

url = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/sales/video_game_sales.csv"

gameSales: DataFrame = spark.read \
    .option("header", "true") \
    .option("sep", "|") \
    .option("inferSchema", "true") \
    .csv(url)

gameSales = gameSales.withColumn("year", F.year("release_date"))

publisherDF = gameSales.filter((F.col("year") >= 2006) & (F.col("year") <= 2015))

bestNAPublisher: str = publisherDF.select("publisher", "na_sales") \
    .groupBy("publisher") \
    .agg(F.sum("na_sales").alias("total_na_sales")) \
    .orderBy(F.desc("total_na_sales")) \
    .first()["publisher"]

# Filter data for the best publisher
publisherDF = publisherDF.filter(F.col("publisher") == bestNAPublisher)

# Count titles with missing North America sales
titlesWithMissingSalesData: int = publisherDF.filter(F.col("na_sales").isNull()).count()

# Aggregate sales data by year
bestNAPublisherSales: DataFrame = publisherDF.groupBy("year") \
    .agg(
        F.round(F.sum("na_sales"), 2).alias('na_total'),
        F.round(F.sum("total_sales"), 2).alias('global_total')
    ) \
    .orderBy("year")

print(f"The publisher with the highest total video game sales in North America is: '{bestNAPublisher}'")
print(f"The number of titles with missing sales data for North America: {titlesWithMissingSalesData}")
print("Sales data for the publisher:")
bestNAPublisherSales.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 2 - Event data from football matches
# MAGIC
# MAGIC A parquet file in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) at folder `assignment/football/events.parquet` based on [https://figshare.com/collections/Soccer_match_event_dataset/4415000/5](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5) contains information about events in [football](https://en.wikipedia.org/wiki/Association_football) matches during the season 2017-18 in five European top-level leagues: English Premier League, Italian Serie A, Spanish La Liga, German Bundesliga, and French Ligue 1.
# MAGIC
# MAGIC #### Background information
# MAGIC
# MAGIC In the considered leagues, a season is played in a double round-robin format where each team plays against all other teams twice. Once as a home team in their own stadium and once as an away team in the other team's stadium. A season usually starts in August and ends in May.
# MAGIC
# MAGIC Each league match consists of two halves of 45 minutes each. Each half runs continuously, meaning that the clock is not stopped when the ball is out of play. The referee of the match may add some additional time to each half based on game stoppages. \[[https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time](https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time)\]
# MAGIC
# MAGIC The team that scores more goals than their opponent wins the match.
# MAGIC
# MAGIC **Columns in the data**
# MAGIC
# MAGIC Each row in the given data represents an event in a specific match. An event can be, for example, a pass, a foul, a shot, or a save attempt.
# MAGIC
# MAGIC Simple explanations for the available columns. Not all of these will be needed in this assignment.
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | competition | string | The name of the competition |
# MAGIC | season | string | The season the match was played |
# MAGIC | matchId | integer | A unique id for the match |
# MAGIC | eventId | integer | A unique id for the event |
# MAGIC | homeTeam | string | The name of the home team |
# MAGIC | awayTeam | string | The name of the away team |
# MAGIC | event | string | The main category for the event |
# MAGIC | subEvent | string | The subcategory for the event |
# MAGIC | eventTeam | string | The name of the team that initiated the event |
# MAGIC | eventPlayerId | integer | The id for the player who initiated the event |
# MAGIC | eventPeriod | string | `1H` for events in the first half, `2H` for events in the second half |
# MAGIC | eventTime | double | The event time in seconds counted from the start of the half |
# MAGIC | tags | array of strings | The descriptions of the tags associated with the event |
# MAGIC | startPosition | struct | The event start position given in `x` and `y` coordinates in range \[0,100\] |
# MAGIC | enPosition | struct | The event end position given in `x` and `y` coordinates in range \[0,100\] |
# MAGIC
# MAGIC The used event categories can be seen from `assignment/football/metadata/eventid2name.csv`.<br>
# MAGIC And all available tag descriptions from `assignment/football/metadata/tags2name.csv`.<br>
# MAGIC You don't need to access these files in the assignment, but they can provide context for the following basic tasks that will use the event data.
# MAGIC
# MAGIC #### The task
# MAGIC
# MAGIC In this task you should load the data with all the rows into a data frame. This data frame object will then be used in the following basic tasks 3-8.

# COMMAND ----------

url = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/events.parquet"
eventDF: DataFrame = spark.read\
    .option("header", "true")\
    .option("sep", "|")\
    .option("inferSchema", "true")\
    .parquet(url)


# eventDF.printSchema()
# eventDF.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 3 - Calculate match results
# MAGIC
# MAGIC Create a match data frame for all the matches included in the event data frame created in basic task 2.
# MAGIC
# MAGIC The resulting data frame should contain one row for each match and include the following columns:
# MAGIC
# MAGIC | column name   | column type | description |
# MAGIC | ------------- | ----------- | ----------- |
# MAGIC | matchId       | integer     | A unique id for the match |
# MAGIC | competition   | string      | The name of the competition |
# MAGIC | season        | string      | The season the match was played |
# MAGIC | homeTeam      | string      | The name of the home team |
# MAGIC | awayTeam      | string      | The name of the away team |
# MAGIC | homeTeamGoals | integer     | The number of goals scored by the home team |
# MAGIC | awayTeamGoals | integer     | The number of goals scored by the away team |
# MAGIC
# MAGIC The number of goals scored for each team should be determined by the available event data.<br>
# MAGIC There are two events related to each goal:
# MAGIC
# MAGIC - One event for the player that scored the goal. This includes possible own goals.
# MAGIC - One event for the goalkeeper that tried to stop the goal.
# MAGIC
# MAGIC You need to choose which types of events you are counting.<br>
# MAGIC If you count both of the event types mentioned above, you will get double the amount of actual goals.

# COMMAND ----------

# Accurate events
filteredDF = eventDF.filter(F.array_contains(F.col("tags"), "Accurate"))

# Add goal columns with intermediate computations
goalsDF = filteredDF.withColumn(
    "isGoal", F.array_contains(F.col("tags"), "Goal")
).withColumn(
    "isOwnGoal", F.array_contains(F.col("tags"), "Own goal")
).withColumn(
    "homeTeamGoals",
    F.when((F.col("isGoal") & (F.col("eventTeam") == F.col("homeTeam"))) |
           (F.col("isOwnGoal") & (F.col("eventTeam") == F.col("awayTeam"))), 1)
    .otherwise(0)
).withColumn(
    "awayTeamGoals",
    F.when((F.col("isGoal") & (F.col("eventTeam") == F.col("awayTeam"))) |
           (F.col("isOwnGoal") & (F.col("eventTeam") == F.col("homeTeam"))), 1)
    .otherwise(0)
)

# Aggregate goals by match
matchDF = goalsDF.groupBy(
    "matchId", "competition", "season", "homeTeam", "awayTeam"
).agg(
    F.sum("homeTeamGoals").alias("homeTeamGoals"),
    F.sum("awayTeamGoals").alias("awayTeamGoals")
)

# Compute and print statistics
matchStats = matchDF.agg(
    F.max(F.col("homeTeamGoals") + F.col("awayTeamGoals")).alias("most_goals"),
    F.sum(F.col("homeTeamGoals") + F.col("awayTeamGoals")).alias("total_goals")
).collect()[0]

print(f"Total number of matches: {matchDF.count()}")
print(f"Matches without any goals: {matchDF.filter((F.col('homeTeamGoals') == 0) & (F.col('awayTeamGoals') == 0)).count()}")
print(f"Most goals in total in a single game: {matchStats['most_goals']}")
print(f"Total amount of goals: {matchStats['total_goals']}")

matchDF.show(10)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 4 - Calculate team points in a season
# MAGIC
# MAGIC Create a season data frame that uses the match data frame from the basic task 3 and contains aggregated seasonal results and statistics for all the teams in all leagues. While the used dataset only includes data from a single season for each league, the code should be written such that it would work even if the data would include matches from multiple seasons for each league.
# MAGIC
# MAGIC ###### Game result determination
# MAGIC
# MAGIC - Team wins the match if they score more goals than their opponent.
# MAGIC - The match is considered a draw if both teams score equal amount of goals.
# MAGIC - Team loses the match if they score fewer goals than their opponent.
# MAGIC
# MAGIC ###### Match point determination
# MAGIC
# MAGIC - The winning team gains 3 points from the match.
# MAGIC - Both teams gain 1 point from a drawn match.
# MAGIC - The losing team does not gain any points from the match.
# MAGIC
# MAGIC The resulting data frame should contain one row for each team per league and season. It should include the following columns:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | competition    | string      | The name of the competition |
# MAGIC | season         | string      | The season |
# MAGIC | team           | string      | The name of the team |
# MAGIC | games          | integer     | The number of games the team played in the given season |
# MAGIC | wins           | integer     | The number of wins the team had in the given season |
# MAGIC | draws          | integer     | The number of draws the team had in the given season |
# MAGIC | losses         | integer     | The number of losses the team had in the given season |
# MAGIC | goalsScored    | integer     | The total number of goals the team scored in the given season |
# MAGIC | goalsConceded  | integer     | The total number of goals scored against the team in the given season |
# MAGIC | points         | integer     | The total number of points gained by the team in the given season |

# COMMAND ----------

# HomeTeam stats 
home_stats = matchDF.select(
    F.col("competition"),
    F.col("season"),
    F.col("homeTeam").alias("team"),
    F.col("homeTeamGoals").alias("goalsScored"),
    F.col("awayTeamGoals").alias("goalsConceded"),
    F.when(F.col("goalsScored") > F.col("goalsConceded"), F.lit(3))
    .when(F.col("goalsScored") == F.col("goalsConceded"), F.lit(1))
    .otherwise(F.lit(0)).alias("points"),
    F.when(F.col("goalsScored") > F.col("goalsConceded"), F.lit(1)).alias("wins"),
    F.when(F.col("goalsScored") == F.col("goalsConceded"), F.lit(1)).alias("draws"),
    F.when(F.col("goalsScored") < F.col("goalsConceded"), F.lit(1)).alias("losses"),
)

# AwayTeam stats
away_stats = matchDF.select(
    F.col("competition"),
    F.col("season"),
    F.col("awayTeam").alias("team"),
    F.col("awayTeamGoals").alias("goalsScored"),
    F.col("homeTeamGoals").alias("goalsConceded"),
    F.when(F.col("goalsScored") > F.col("goalsConceded"), F.lit(3))
    .when(F.col("goalsScored") == F.col("goalsConceded"), F.lit(1))
    .otherwise(F.lit(0)).alias("points"),
    F.when(F.col("goalsScored") > F.col("goalsConceded"), F.lit(1)).alias("wins"),
    F.when(F.col("goalsScored") == F.col("goalsConceded"), F.lit(1)).alias("draws"),
    F.when(F.col("goalsScored") < F.col("goalsConceded"), F.lit(1)).alias("losses"),
)

all_stats = home_stats.union(away_stats)

# Aggregiate the season data frame
seasonDF: DataFrame = (
    all_stats.groupBy("competition", "season", "team")
    .agg(
        F.sum("goalsScored").alias("goalsScored"),
        F.sum("goalsConceded").alias("goalsConceded"),
        F.sum("points").alias("points"),
        F.sum("wins").alias("wins"),
        F.sum("draws").alias("draws"),
        F.sum("losses").alias("losses"),
        F.count("team").alias("games"),
    )
)

print(f"Total number of rows: {seasonDF.count()}")
print(f"Teams with more than 70 points in a season: {seasonDF.filter(F.col('points') > 70).count()}")
print(f"Lowest amount points in a season: {seasonDF.agg(F.min('points')).collect()[0][0]}")
print(f"Total amount of points: {seasonDF.agg(F.sum('points')).collect()[0][0]}")
print(f"Total amount of goals scored: {seasonDF.agg(F.sum('goalsScored')).collect()[0][0]}")
print(f"Total amount of goals conceded: {seasonDF.agg(F.sum('goalsConceded')).collect()[0][0]}")
seasonDF.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 5 - English Premier League table
# MAGIC
# MAGIC Using the season data frame from basic task 4 calculate the final league table for `English Premier League` in season `2017-2018`.
# MAGIC
# MAGIC The result should be given as data frame which is ordered by the team's classification for the season.
# MAGIC
# MAGIC A team is classified higher than the other team if one of the following is true:
# MAGIC
# MAGIC - The team has a higher number of total points than the other team
# MAGIC - The team has an equal number of points, but have a better goal difference than the other team
# MAGIC - The team has an equal number of points and goal difference, but have more goals scored in total than the other team
# MAGIC
# MAGIC Goal difference is the difference between the number of goals scored for and against the team.
# MAGIC
# MAGIC The resulting data frame should contain one row for each team.<br>
# MAGIC It should include the following columns (several columns renamed trying to match the [league table in Wikipedia](https://en.wikipedia.org/wiki/2017%E2%80%9318_Premier_League#League_table)):
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | Pos         | integer     | The classification of the team |
# MAGIC | Team        | string      | The name of the team |
# MAGIC | Pld         | integer     | The number of games played |
# MAGIC | W           | integer     | The number of wins |
# MAGIC | D           | integer     | The number of draws |
# MAGIC | L           | integer     | The number of losses |
# MAGIC | GF          | integer     | The total number of goals scored by the team |
# MAGIC | GA          | integer     | The total number of goals scored against the team |
# MAGIC | GD          | string      | The goal difference |
# MAGIC | Pts         | integer     | The total number of points gained by the team |
# MAGIC
# MAGIC The goal difference should be given as a string with an added `+` at the beginning if the difference is positive, similarly to the table in the linked Wikipedia article.

# COMMAND ----------

season = "2017-2018"
englandDF: DataFrame = seasonDF.filter(F.col("competition") == "English Premier League")
englandDF = englandDF.filter(F.col("season") == season)
englandDF = englandDF.withColumn("goalDiff", F.col("goalsScored") - F.col("goalsConceded"))
englandDF = englandDF.orderBy(F.col("points").desc(), F.col("goalDiff").desc(), F.col("goalsScored").desc())
englandDF = englandDF.withColumn("Pld", F.col("wins") + F.col("draws") + F.col("losses"))
englandDF = englandDF.select(
  F.col("team").alias('Team'), 
  F.col("Pld"), 
  F.col("wins").alias("W"),
  F.col("draws").alias("D"),
  F.col("losses").alias("L"),
  F.col("goalsScored").alias("GF"),
  F.col("goalsConceded").alias("GA"),
  F.col("goalDiff").alias("GD"),
  F.col("points").alias("Pts")
  )

englandDF = englandDF.rdd.zipWithIndex().toDF(["data", "index"])
englandDF = englandDF.select((F.col("index")+1).alias("Pos"), F.col("data.*"))

print(f"English Premier League table for season {season}")
englandDF.show(20, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic task 6: Calculate the number of passes
# MAGIC
# MAGIC This task involves going back to the event data frame and counting the number of passes each team made in each match. A pass is considered successful if it is marked as `Accurate`.
# MAGIC
# MAGIC Using the event data frame from basic task 2, calculate the total number of passes as well as the total number of successful passes for each team in each match.<br>
# MAGIC The resulting data frame should contain one row for each team in each match, i.e., two rows for each match. It should include the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | matchId     | integer     | A unique id for the match |
# MAGIC | competition | string      | The name of the competition |
# MAGIC | season      | string      | The season |
# MAGIC | team        | string      | The name of the team |
# MAGIC | totalPasses | integer     | The total number of passes the team attempted in the match |
# MAGIC | successfulPasses | integer | The total number of successful passes made by the team in the match |
# MAGIC
# MAGIC You can assume that each team had at least one pass attempt in each match they played.

# COMMAND ----------

matchPassDF: DataFrame = eventDF.select("matchId", "competition", "season", "homeTeam", "awayTeam", "eventTeam", "tags")

matchPassDF = matchPassDF.filter(F.col("event") == "Pass")

# Process home team passes
homeTeamPassDF = matchPassDF.withColumn("pass", 
                             F.when(F.col("eventTeam") == F.col("homeTeam"), 1)\
                                 .otherwise(0))

homeTeamPassDF = homeTeamPassDF.withColumn("passSuccess", 
                             F.when((F.col("eventTeam") == F.col("homeTeam")) & (F.array_contains(F.col("tags"), "Accurate")), 1)\
                                 .otherwise(0))

homeTeamPassDF = homeTeamPassDF.withColumnRenamed("homeTeam", "team")

# Process away team passes
awayTeamPassDF = matchPassDF.withColumn("pass", 
                             F.when(F.col("eventTeam") == F.col("awayTeam"), 1)\
                                 .otherwise(0))

awayTeamPassDF = awayTeamPassDF.withColumn("passSuccess", 
                             F.when((F.col("eventTeam") == F.col("awayTeam")) & (F.array_contains(F.col("tags"), "Accurate")), 1)\
                                 .otherwise(0))

awayTeamPassDF = awayTeamPassDF.withColumnRenamed("awayTeam", "team")

# Group by match and team
homeTeamPassDF = homeTeamPassDF.groupBy("matchId", "competition", "season", "team").agg(F.sum("pass").alias("totalPasses"), F.sum("passSuccess").alias("successfulPasses"))

awayTeamPassDF = awayTeamPassDF.groupBy("matchId", "competition", "season", "team").agg(F.sum("pass").alias("totalPasses"), F.sum("passSuccess").alias("successfulPasses"))

matchPassDF = homeTeamPassDF.union(awayTeamPassDF)

print(f"Total number of rows: {matchPassDF.count()}")
print(f"Team-match pairs with more than 700 total passes: {matchPassDF.filter(F.col('totalPasses') > 700).count()}")
print(f"Team-match pairs with more than 600 successful passes: {matchPassDF.filter(F.col('successfulPasses') > 600).count()}")

matchPassDF.show(6)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 7: Teams with the worst passes
# MAGIC
# MAGIC Using the match pass data frame from basic task 6 find the teams with the lowest average ratio for successful passes over the season `2017-2018` for each league.
# MAGIC
# MAGIC The ratio for successful passes over a single match is the number of successful passes divided by the number of total passes.<br>
# MAGIC The average ratio over the season is the average of the single match ratios.
# MAGIC
# MAGIC Give the result as a data frame that has one row for each league-team pair with the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | competition | string      | The name of the competition |
# MAGIC | team        | string      | The name of the team |
# MAGIC | passSuccessRatio | double | The average ratio for successful passes over the season given as percentages rounded to two decimals |
# MAGIC
# MAGIC Order the data frame so that the team with the lowest ratio for passes is given first.

# COMMAND ----------

# The ratio for successful passes over a single match
lowestPassSuccessRatioDF: DataFrame = matchPassDF.withColumn("passSuccessRatio", F.col("successfulPasses") / F.col("totalPasses"))

lowestPassSuccessRatioDF = lowestPassSuccessRatioDF.groupBy("competition", "season", "team").agg(F.mean("passSuccessRatio").alias("passSuccessRatio"))

# Window Function for Ranking
window_spec = Window.partitionBy('competition').orderBy(F.col('passSuccessRatio').asc())

lowestPassSuccessRatioDF = (
    lowestPassSuccessRatioDF.withColumn('row_number', F.row_number().over(window_spec))
      .select('competition', 'team', 'passSuccessRatio')
)

lowestPassSuccessRatioDF = lowestPassSuccessRatioDF.sort(F.col("passSuccessRatio").asc())
lowestPassSuccessRatioDF = lowestPassSuccessRatioDF.withColumn("passSuccessRatio", F.round(100*F.col("passSuccessRatio"), 2))

print("Teams with the lowest ratios for successful passes for each league in season 2017-2018:")
lowestPassSuccessRatioDF.show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic task 8: The best teams
# MAGIC
# MAGIC For this task the best teams are determined by having the highest point average per match.
# MAGIC
# MAGIC Using the data frames created in the previous tasks find the two best teams from each league in season `2017-2018` with their full statistics.
# MAGIC
# MAGIC Give the result as a data frame with the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | Team        | string      | The name of the team |
# MAGIC | League      | string      | The name of the league |
# MAGIC | Pos         | integer     | The classification of the team within their league |
# MAGIC | Pld         | integer     | The number of games played |
# MAGIC | W           | integer     | The number of wins |
# MAGIC | D           | integer     | The number of draws |
# MAGIC | L           | integer     | The number of losses |
# MAGIC | GF          | integer     | The total number of goals scored by the team |
# MAGIC | GA          | integer     | The total number of goals scored against the team |
# MAGIC | GD          | string      | The goal difference |
# MAGIC | Pts         | integer     | The total number of points gained by the team |
# MAGIC | Avg         | double      | The average points per match gained by the team |
# MAGIC | PassRatio   | double      | The average ratio for successful passes over the season given as percentages rounded to two decimals |
# MAGIC
# MAGIC Order the data frame so that the team with the highest point average per match is given first.

# COMMAND ----------

# Filter the columns to be displayed
bestDF: DataFrame = seasonDF.filter(F.col("season") == "2017-2018") \
    .withColumn("Avg", F.col("points") / F.col("games"))

# Window Function for Ranking
window_spec = Window.partitionBy('competition').orderBy(F.col('points').desc())
bestDF = bestDF.withColumn('Pos', F.row_number().over(window_spec)).filter(F.col('Pos') <= 2)

bestDF = bestDF.join(lowestPassSuccessRatioDF, ['competition', 'team'], 'left')

bestDF = bestDF.select(
    F.col("team").alias('Team'),
    F.col("competition").alias('League'),
    F.col("Pos"),
    F.col("games").alias('Pld'),
    F.col("wins").alias('W'),
    F.col("draws").alias('D'),
    F.col("losses").alias('L'),
    F.col("goalsScored").alias('GF'),
    F.col("goalsConceded").alias('GA'),
    (F.col("GF") - F.col("GA")).alias('GD'),
    F.col("points").alias('Pts'),
    F.round(F.col("Avg"), 2).alias('Avg'),
    F.col("passSuccessRatio").alias('PassRatio')
)

bestDF = bestDF.sort(F.col('Pos').asc(), F.col('Avg').desc())

print("The top 2 teams for each league in season 2017-2018")
bestDF.show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 9: General information
# MAGIC
# MAGIC Answer **briefly** to the following questions.
# MAGIC
# MAGIC Remember that using AI and collaborating with other students outside your group is allowed as long as the usage and collaboration is documented.<br>
# MAGIC However, every member of the group should have some contribution to the assignment work.
# MAGIC
# MAGIC - Who were your group members and their contributions to the work?
# MAGIC     - Solo groups can ignore this question.
# MAGIC
# MAGIC - Did you use AI tools while doing the assignment?
# MAGIC     - Which ones and how did they help?
# MAGIC
# MAGIC - Did you work with students outside your assignment group?
# MAGIC     - Who or which group? (only extensive collaboration need to reported)

# COMMAND ----------

# MAGIC %md
# MAGIC * <h4>Group Members: Maral Nourimand & Ali Jedari Heidarzadeh</h4>
# MAGIC
# MAGIC   **Maral's contributions:**
# MAGIC   * Even numbered basic tasks
# MAGIC   * Advanced tasks 1 & 2
# MAGIC   * Cross-checking the results
# MAGIC
# MAGIC   **Ali's contributions:**
# MAGIC   * Odd numbered basic tasks
# MAGIC   * Advanced task 4
# MAGIC   * Debugging & Result improvement
# MAGIC
# MAGIC * We used ChatGPT to find inspirations about how to do some tasks, and the Databricks Assistant to correct the mistakes when there were errors and exceptions in the code.
# MAGIC
# MAGIC * There was no collaboration with other students.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced tasks
# MAGIC
# MAGIC The implementation of the basic tasks is compulsory for every group.
# MAGIC
# MAGIC Doing the following advanced tasks you can gain course points which can help in getting a better grade from the course.<br>
# MAGIC Partial solutions can give partial points.
# MAGIC
# MAGIC The advanced task 1 will be considered in the grading for every group based on their solutions for the basic tasks.
# MAGIC
# MAGIC The advanced tasks 2, 3, and 4 are separate tasks. The solutions used in these other advanced tasks do not affect the grading of advanced task 1. Instead, a good use of optimized methods can positively influence the grading of each specific task, while very non-optimized solutions can have a negative effect on the task grade.
# MAGIC
# MAGIC While you can attempt all three tasks (advanced tasks 2-4), only two of them will be graded and contribute towards the course grade.<br>
# MAGIC Mark in the following cell which tasks you want to be graded and which should be ignored.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### If you did the advanced tasks 2-4, mark here which of the two should be considered in grading:
# MAGIC
# MAGIC - Advanced task 2 should be graded: Done
# MAGIC - Advanced task 3 should be graded: ???
# MAGIC - Advanced task 4 should be graded: Done

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 1 - Optimized and correct solutions to the basic tasks (2 points)
# MAGIC
# MAGIC Use the tools Spark offers effectively and avoid unnecessary operations in the code for the basic tasks.
# MAGIC
# MAGIC A couple of things to consider (**not** even close to a complete list):
# MAGIC
# MAGIC - Consider using explicit schemas when dealing with CSV data sources.
# MAGIC - Consider only including those columns from a data source that are actually needed.
# MAGIC - Filter unnecessary rows whenever possible to get smaller datasets.
# MAGIC - Avoid collect or similar expensive operations for large datasets.
# MAGIC - Consider using explicit caching if some data frame is used repeatedly.
# MAGIC - Avoid unnecessary shuffling (for example sorting) operations.
# MAGIC - Avoid unnecessary actions (count, etc.) that are not needed for the task.
# MAGIC
# MAGIC In addition to the effectiveness of your solutions, the correctness of the solution logic will be taken into account when determining the grade for this advanced task 1.
# MAGIC "A close enough" solution with some logic fails might be enough to have an accepted group assignment, but those failings might lower the score for this task.
# MAGIC
# MAGIC It is okay to have your own test code that would fall into category of "ineffective usage" or "unnecessary operations" while doing the assignment tasks. However, for the final Moodle submission you should comment out or delete such code (and test that you have not broken anything when doing the final modifications).
# MAGIC
# MAGIC Note, that you should not do the basic tasks again for this additional task, but instead modify your basic task code with more efficient versions.
# MAGIC
# MAGIC You can create a text cell below this one and describe what optimizations you have done. This might help the grader to better recognize how skilled your work with the basic tasks has been.

# COMMAND ----------

# MAGIC %md
# MAGIC We improved the code as we went through it once more after the generation. Here are some of the changes made and the reasoning behind them, considering each task.  
# MAGIC
# MAGIC
# MAGIC Task1:
# MAGIC - Reduced Shuffling: By filtering and aggregating earlier, unnecessary data movement is minimized. Hence, we reduced shuffling. 
# MAGIC - First, we used collect() to retrieve the data. Replacing collect() with first() avoids expensive driver-side operations.
# MAGIC
# MAGIC Task3:
# MAGIC We extract the “Goal” and the “Own goal” tag to deal with the match results.
# MAGIC - Two columns of isGoal and isOwnGoal are computed once and reused in both homeTeamGoals and awayTeamGoals Datafreames.
# MAGIC - We tried to reduce DataFrame overwrites by chaining operations (withColumn and agg).
# MAGIC - We filtered rows with F.array_contains("tags", "Accurate") early to minimize unnecessary processing.
# MAGIC - We also Avoided Redundant Data Collection Using distributed aggregation (agg) instead of multiple collect() calls to compute most_goals and total_goals.
# MAGIC
# MAGIC Task4:
# MAGIC - We filtered out the columns not needed and used F.when() to avoid redundancy of copying multiple Dataframes. Something considered in the previous tasks, too.
# MAGIC
# MAGIC Task 5:
# MAGIC - Chained Filtering allows Spark to prune irrelevant data early in the query plan, minimizing I/O and memory overhead. It means that filters are applied sequentially (competition == "English Premier League" and season == "2017-2018").
# MAGIC - New columns such as goalDiff and Pld are computed directly using withColumn. This avoids creating intermediate DataFrames unnecessarily and keeps transformations succinct. So, we Derived Columns Inline.
# MAGIC
# MAGIC Task 6:
# MAGIC - The use of union efficiently combines the results for home and away teams into a single DataFrame. Since both DataFrames have the same schema, union avoids reshuffling, making it a straightforward and efficient operation.
# MAGIC - We could have used struct instead of using withColumn() multiple time but for 2 or 3 times its effect is negligible. So, for the sake of clarity and readability we kept withColumn().
# MAGIC
# MAGIC Task 7:
# MAGIC - The use of Window.partitionBy().orderBy() and row_number() efficiently ranks teams by their passSuccessRatio within each competition. This is much more efficient than sorting the entire dataset at once because it avoids a global sort.
# MAGIC - Sorting is done after ranking, avoiding unnecessary sorting before aggregation.
# MAGIC
# MAGIC Task 8:
# MAGIC - Same as what we did in the previous tasks, transformation using withColumn() and window Function for Ranking were applied here, too.
# MAGIC - We used a left join between bestDF and lowestPassSuccessRatioDF on competition and team. The join is performed on the necessary keys, which ensures that Spark can use its optimized join strategies.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 2 - Further tasks with football data (2 points)
# MAGIC
# MAGIC This advanced task continues with football event data from the basic tasks. In addition, there are two further related datasets that are used in this task.
# MAGIC
# MAGIC A Parquet file at folder `assignment/football/matches.parquet` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains information about which players were involved on each match including information on the substitutions made during the match.
# MAGIC
# MAGIC Another Parquet file at folder `assignment/football/players.parquet` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains information about the player names, default roles when playing, and their birth areas.
# MAGIC
# MAGIC #### Columns in the additional data
# MAGIC
# MAGIC The match dataset (`assignment/football/matches.parquet`) has one row for each match and each row has the following columns:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | matchId      | integer     | A unique id for the match |
# MAGIC | competition  | string      | The name of the league |
# MAGIC | season       | string      | The season the match was played |
# MAGIC | roundId      | integer     | A unique id for the round in the competition |
# MAGIC | gameWeek     | integer     | The gameWeek of the match |
# MAGIC | date         | date        | The date the match was played |
# MAGIC | status       | string      | The status of the match, `Played` if the match has been played |
# MAGIC | homeTeamData | struct      | The home team data, see the table below for the attributes in the struct |
# MAGIC | awayTeamData | struct      | The away team data, see the table below for the attributes in the struct |
# MAGIC | referees     | struct      | The referees for the match |
# MAGIC
# MAGIC Both team data columns have the following inner structure:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | team         | string      | The name of the team |
# MAGIC | coachId      | integer     | A unique id for the coach of the team |
# MAGIC | lineup       | array of integers | A list of the player ids who start the match on the field for the team |
# MAGIC | bench        | array of integers | A list of the player ids who start the match on the bench, i.e., the reserve players for the team |
# MAGIC | substitution1 | struct     | The first substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC | substitution2 | struct     | The second substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC | substitution3 | struct     | The third substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC
# MAGIC Each substitution structs have the following inner structure:
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | playerIn     | integer     | The id for the player who was substituted from the bench into the field, i.e., this player started playing after this substitution |
# MAGIC | playerOut    | integer     | The id for the player who was substituted from the field to the bench, i.e., this player stopped playing after this substitution |
# MAGIC | minute       | integer     | The minute from the start of the match the substitution was made.<br>Values of 45 or less indicate that the substitution was made in the first half of the match,<br>and values larger than 45 indicate that the substitution was made on the second half of the match. |
# MAGIC
# MAGIC The player dataset (`assignment/football/players.parquet`) has the following columns:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | playerId     | integer     | A unique id for the player |
# MAGIC | firstName    | string      | The first name of the player |
# MAGIC | lastName     | string      | The last name of the player |
# MAGIC | birthArea    | string      | The birth area (nation or similar) of the player |
# MAGIC | role         | string      | The main role of the player, either `Goalkeeper`, `Defender`, `Midfielder`, or `Forward` |
# MAGIC | foot         | string      | The stronger foot of the player |
# MAGIC
# MAGIC #### Background information
# MAGIC
# MAGIC In a football match both teams have 11 players on the playing field or pitch at the start of the match. Each team also have some number of reserve players on the bench at the start of the match. The teams can make up to three substitution during the match where they switch one of the players on the field to a reserve player. (Currently, more substitutions are allowed, but at the time when the data is from, three substitutions were the maximum.) Any player starting the match as a reserve and who is not substituted to the field during the match does not play any minutes and are not considered involved in the match.
# MAGIC
# MAGIC For this task the length of each match should be estimated with the following procedure:
# MAGIC
# MAGIC - Only the additional time added to the second half of the match should be considered. I.e., the length of the first half is always considered to be 45 minutes.
# MAGIC - The length of the second half is to be considered as the last event of the half rounded upwards towards the nearest minute.
# MAGIC     - I.e., if the last event of the second half happens at 2845 seconds (=47.4 minutes) from the start of the half, the length of the half should be considered as 48 minutes. And thus, the full length of the entire match as 93 minutes.
# MAGIC
# MAGIC A personal plus-minus statistics for each player can be calculated using the following information:
# MAGIC
# MAGIC - If a goal was scored by the player's team when the player was on the field, `add 1`
# MAGIC - If a goal was scored by the opponent's team when the player was on the field, `subtract 1`
# MAGIC - If a goal was scored when the player was a reserve on the bench, `no change`
# MAGIC - For any event that is not a goal, or is in a match that the player was not involved in, `no change`
# MAGIC - Any substitutions is considered to be done at the start of the given minute.
# MAGIC     - I.e., if the player is substituted from the bench to the field at minute 80 (minute 35 on the second half), they were considered to be on the pitch from second 2100.0 on the 2nd half of the match.
# MAGIC - If a goal was scored in the additional time of the first half of the match, i.e., the goal event period is `1H` and event time is larger than 2700 seconds, some extra considerations should be taken into account:
# MAGIC     - If a player is substituted into the field at the beginning of the second half, `no change`
# MAGIC     - If a player is substituted off the field at the beginning of the second half, either `add 1` or `subtract 1` depending on team that scored the goal
# MAGIC     - Any player who is substituted into the field at minute 45 or later is only playing on the second half of the match.
# MAGIC     - Any player who is substituted off the field at minute 45 or later is considered to be playing the entire first half including the additional time.
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC The target of the task is to use the football event data and the additional datasets to determine the following:
# MAGIC
# MAGIC - The players with the most total minutes played in season 2017-2018 for each player role
# MAGIC     - I.e., the player in Goalkeeper role who has played the longest time across all included leagues. And the same for the other player roles (Defender, Midfielder, and Forward)
# MAGIC     - Give the result as a data frame that has the following columns:
# MAGIC         - `role`: the player role
# MAGIC         - `player`: the full name of the player, i.e., the first name combined with the last name
# MAGIC         - `birthArea`: the birth area of the player
# MAGIC         - `minutes`: the total minutes the player played during season 2017-2018
# MAGIC - The players with higher than `+65` for the total plus-minus statistics in season 2017-2018
# MAGIC     - Give the result as a data frame that has the following columns:
# MAGIC         - `player`: the full name of the player, i.e., the first name combined with the last name
# MAGIC         - `birthArea`: the birth area of the player
# MAGIC         - `role`: the player role
# MAGIC         - `plusMinus`: the total plus-minus statistics for the player during season 2017-2018
# MAGIC
# MAGIC It is advisable to work towards the target results using several intermediate steps.

# COMMAND ----------

# Load the data
url = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/matches.parquet"
matchesDF: DataFrame = spark.read\
    .option("header", "true")\
    .option("sep", "|")\
    .option("inferSchema", "true")\
    .parquet(url)

# matchesDF.printSchema()
# matchesDF.show(5)

url = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/players.parquet"
playersDF: DataFrame = spark.read\
    .option("header", "true")\
    .option("sep", "|")\
    .option("inferSchema", "true")\
    .parquet(url)

# playersDF.printSchema()
# playersDF.show(5)


# COMMAND ----------

# Group events by matchId and eventPeriod
last_events_per_match = (
    eventDF.groupBy("matchId", "eventPeriod")
    .agg(F.max("eventTime").alias("lastEventTime"))
)

# Add a column for half duration (1H is fixed at 45 minutes; 2H is rounded up from seconds to minutes)
last_events_per_match = last_events_per_match.withColumn(
    "halfDuration",
    F.when(F.col("eventPeriod") == "1H", 45).otherwise(F.ceil(F.col("lastEventTime") / 60))
)

# Sum the durations for each matchId to get the total match duration
match_durations = (
    last_events_per_match.groupBy("matchId")
    .agg(F.sum("halfDuration").alias("matchLength"))
)

# chech with the output examples to see if it works correctly 
# match_durations.filter(F.col('matchId') == 2565891).show(truncate=False)

# match_durations.show(10)

# COMMAND ----------

# Helper function to process players on the field and bench
def process_team_data_with_bench(df, column_prefix):
    # Lineup players: start at 0, end determined later
    lineup = df.select(
        F.col("matchId"),
        F.col("competition"),
        F.col("season"),
        F.col(f"{column_prefix}team").alias("playerTeam"),
        F.explode(F.col(f"{column_prefix}lineup")).alias("playerId"),
        F.lit(0).alias("startMinute"),
        F.lit(None).cast("double").alias("endMinute")
    )

    # Substitution players (incoming): extract start minutes
    substitutions_in = [
        df.select(
            F.col("matchId"),
            F.col("competition"),
            F.col("season"),
            F.col(f"{column_prefix}team").alias("playerTeam"),
            F.col(f"{column_prefix}substitution{i}.playerIn").alias("playerId"),
            F.col(f"{column_prefix}substitution{i}.minute").alias("startMinute"),
            F.lit(None).cast("double").alias("endMinute")
        )
        for i in range(1, 4)
    ]
    substitutions_in_combined = reduce(lambda df1, df2: df1.union(df2), substitutions_in)

    # Substitution players (outgoing): extract end minutes
    substitutions_out = [
        df.select(
            F.col("matchId"),
            F.col("competition"),
            F.col("season"),
            F.col(f"{column_prefix}team").alias("playerTeam"),
            F.col(f"{column_prefix}substitution{i}.playerOut").alias("playerId"),
            F.lit(None).cast("double").alias("startMinute"),
            F.col(f"{column_prefix}substitution{i}.minute").alias("endMinute")
        )
        for i in range(1, 4)
    ]
    substitutions_out_combined = reduce(lambda df1, df2: df1.union(df2), substitutions_out)

    # Bench players who never enter the match: duration is 0
    bench = df.select(
        F.col("matchId"),
        F.col("competition"),
        F.col("season"),
        F.col(f"{column_prefix}team").alias("playerTeam"),
        F.explode(F.col(f"{column_prefix}bench")).alias("playerId"),
        F.lit(None).cast("double").alias("startMinute"),
        F.lit(None).cast("double").alias("endMinute")
    )

    # Combine lineup, incoming subs, outgoing subs, and bench players
    return lineup.unionByName(substitutions_in_combined).unionByName(substitutions_out_combined).unionByName(bench)

# Process home and away teams with benches
home_players = process_team_data_with_bench(matchesDF, "homeTeamData.")
away_players = process_team_data_with_bench(matchesDF, "awayTeamData.")

# Combine all players
all_players = home_players.unionByName(away_players)

# Remove duplicate rows for players who start on the field and are later substituted out
all_players_deduplicated = (
    all_players.groupBy("matchId", "playerId")
    .agg(
        F.first("competition").alias("competition"),
        F.first("season").alias("season"),
        F.first("playerTeam").alias("playerTeam"),
        F.min("startMinute").alias("startMinute"),
        F.max("endMinute").alias("endMinute")
    )
)

# Join with match durations
all_players_with_durations = all_players_deduplicated.join(
    match_durations,
    on="matchId",
    how="left"
).withColumn(
    "endMinute",
    F.when(F.col("endMinute").isNull() & F.col("startMinute").isNotNull(), F.col("matchLength"))
    .otherwise(F.col("endMinute"))
)

# Calculate minutes on the pitch
player_time_on_pitch = all_players_with_durations.withColumn(
    "minutes", 
    F.when(F.col("startMinute").isNotNull(), F.col("endMinute") - F.col("startMinute")).otherwise(0)
).select("matchId", "playerId", "competition", "season", "playerTeam", "startMinute", "endMinute", "minutes")


# player_time_on_pitch.filter(F.col('matchId') == 2575959).filter(F.col('playerId') == 20820).show(truncate=False)
# player_time_on_pitch.show(5)

# COMMAND ----------

# Filter goal events
goal_events = eventDF.filter(
    F.array_contains(F.col("tags"), "Goal") & F.array_contains(F.col("tags"), "Accurate")
).select(
    "matchId", "eventId", "eventTeam", "eventTime", "eventPeriod"
)

# Adjust eventTime to minutes
goal_events = goal_events.withColumn(
    "eventTime",
    F.when(F.col("eventPeriod") == "1H", F.col("eventTime") / 60)
    .when(F.col("eventPeriod") == "2H", 45 + F.col("eventTime") / 60)
)

# Renaming columns for clarity
goal_events = goal_events.withColumnRenamed("matchId", "goal_matchId")

# Cross-join goal events with all players on the pitch
player_events = goal_events.crossJoin(player_time_on_pitch).filter(
    goal_events["goal_matchId"] == player_time_on_pitch["matchId"]
)

# Determine if the player was on the pitch at the time of the goal event
player_events = player_events.filter(
    (player_events["eventTime"] >= player_events["startMinute"]) &
    (player_events["eventTime"] < player_events["endMinute"])
)

# Calculate playerPlusMinus
player_events = player_events.withColumn(
    "playerPlusMinus",
    F.when(
        player_events["playerTeam"] == player_events["eventTeam"],  # Scoring team
        1
    ).when(
        player_events["playerTeam"] != player_events["eventTeam"],  # Opposing team
        -1
    )
)

final_result = player_events.select(
    "goal_matchId", 
    "eventId", 
    "playerId", 
    "playerTeam", 
    "eventTime", 
    "eventTeam",  
    "startMinute", 
    "endMinute", 
    "playerPlusMinus"
).withColumnRenamed("goal_matchId", "matchId")

# final_result.filter(F.col('eventId') == 177960849).filter(F.col('matchId') == 2499719).filter(F.col('playerId') == 8013).show(truncate=False)
# final_result.show(10)


# COMMAND ----------

# Filter for season 2017-2018
season_players = player_time_on_pitch.filter(F.col("season") == "2017-2018")

# Aggregate total minutes per player for the season
total_minutes_per_player = season_players.groupBy("playerId").agg(
    F.sum("minutes").alias("totalMinutes")
)

# Join with playersDF to get player details
mostMinutesDF = total_minutes_per_player.join(
    playersDF,
    on="playerId",
    how="inner"
).select(
    "playerId",
    F.concat(F.col("firstName"), F.lit(" "), F.col("lastName")).alias("player"),
    "birthArea",
    "role",
    "totalMinutes"
)

# Find the player with the most minutes per role
role_window = Window.partitionBy("role").orderBy(F.desc("totalMinutes"))

ranked_players = mostMinutesDF.withColumn("rank", F.row_number().over(role_window))

mostMinutesDF = ranked_players.filter(F.col("rank") == 1).select(
    "role", "player", "birthArea", F.col("totalMinutes").alias("minutes")
)

print("The players with the most minutes played in season 2017-2018 for each player role:")
mostMinutesDF.show(truncate=False)

# COMMAND ----------


#  Aggregate total plus-minus statistics for each player
player_plus_minus = final_result.groupBy("playerId").agg(
    F.sum("playerPlusMinus").alias("plusMinus")
)

topPlayers: DataFrame = player_plus_minus.filter(F.col("plusMinus") > 65)

# Join with playersDF
topPlayers = topPlayers.join(
    playersDF, 
    on="playerId", 
    how="inner"
).select(
    F.concat(F.col("firstName"), F.lit(" "), F.col("lastName")).alias("player"),
    "birthArea",
    "role",
    "plusMinus"
)

print("The players with higher than +65 for the plus-minus statistics in season 2017-2018:")
topPlayers.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 3 - Image data and pixel colors (2 points)
# MAGIC
# MAGIC This advanced task involves loading in PNG image data and complementing JSON metadata into Spark data structure. And then determining the colors of the pixels in the images, and finding the answers to several color related questions.
# MAGIC
# MAGIC The folder `assignment/openmoji/color` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains collection of PNG images from [OpenMoji](https://openmoji.org/) project.
# MAGIC
# MAGIC The JSON Lines formatted file `assignment/openmoji/openmoji.jsonl` contains metadata about the image collection. Only a portion of the images are included as source data for this task, so the metadata file contains also information about images not considered in this task.
# MAGIC
# MAGIC #### Data description and helper functions
# MAGIC
# MAGIC The image data considered in this task can be loaded into a Spark data frame using the `image` format: [https://spark.apache.org/docs/3.5.0/ml-datasource.html](https://spark.apache.org/docs/3.5.0/ml-datasource.html). The resulting data frame contains a single column which includes information about the filename, image size as well as the binary data representing the image itself. The Spark documentation page contains more detailed information about the structure of the column.
# MAGIC
# MAGIC Instead of using the images as source data for machine learning tasks, the binary image data is accessed directly in this task.<br>
# MAGIC You are given two helper functions to help in dealing with the binary data:
# MAGIC
# MAGIC - Function `toPixels` takes in the binary image data and the number channels used to represent each pixel.
# MAGIC     - In the case of the images used in this task, the number of channels match the number bytes used for each pixel.
# MAGIC     - As output the function returns an array of strings where each string is hexadecimal representation of a single pixel in the image.
# MAGIC - Function `toColorName` takes in a single pixel represented as hexadecimal string.
# MAGIC     - As output the function returns a string with the name of the basic color that most closely represents the pixel.
# MAGIC     - The function uses somewhat naive algorithm to determine the name of the color, and does not always give correct results.
# MAGIC     - Many of the pixels in this task have a lot of transparent pixels. Any such pixel is marked as the color `None` by the function.
# MAGIC
# MAGIC With the help of the given functions it is possible to transform the binary image data to an array of color names without using additional libraries or knowing much about image processing.
# MAGIC
# MAGIC The metadata file given in JSON Lines format can be loaded into a Spark data frame using the `json` format: [https://spark.apache.org/docs/3.5.0/sql-data-sources-json.html](https://spark.apache.org/docs/3.5.0/sql-data-sources-json.html). The attributes used in the JSON data are not described here, but are left for you to explore. The original regular JSON formatted file can be found at [https://github.com/hfg-gmuend/openmoji/blob/master/data/openmoji.json](https://github.com/hfg-gmuend/openmoji/blob/master/data/openmoji.json).
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC The target of the task is to combine the image data with the JSON data, determine the image pixel colors, and the find the answers to the following questions:
# MAGIC
# MAGIC - Which four images have the most colored non-transparent pixels?
# MAGIC - Which five images have the lowest ratio of colored vs. transparent pixels?
# MAGIC - What are the three most common colors in the Finnish flag image (annotation: `flag: Finland`)?
# MAGIC     - And how many percentages of the colored pixels does each color have?
# MAGIC - How many images have their most common three colors as, `Blue`-`Yellow`-`Black`, in that order?
# MAGIC - Which five images have the most red pixels among the image group `activities`?
# MAGIC     - And how many red pixels do each of these images have?
# MAGIC
# MAGIC It might be advisable to test your work-in-progress code with a limited number of images before using the full image set.<br>
# MAGIC You are free to choose your own approach to the task: user defined functions with data frames, RDDs/Datasets, or combination of both.

# COMMAND ----------

# separates binary image data to an array of hex strings that represent the pixels
# assumes 8-bit representation for each pixel (0x00 - 0xff)
# with `channels` attribute representing how many bytes is used for each pixel
'''
def toPixels(data: bytes, channels: int) -> List[str]:
    return [
        "".join([
            f"{data[index+byte]:02X}"
            for byte in range(0, channels)
        ])
        for index in range(0, len(data), channels)
    ]
'''

# COMMAND ----------

# naive implementation of picking the name of the pixel color based on the input hex representation of the pixel
# only works for OpenCV type CV_8U (mode=24) compatible input
'''
def toColorName(hexString: str) -> str:
    # mapping of RGB values to basic color names
    colors: Dict[Tuple[int, int, int], str] = {
        (0, 0, 0):     "Black",  (0, 0, 128):     "Blue",   (0, 0, 255):     "Blue",
        (0, 128, 0):   "Green",  (0, 128, 128):   "Green",  (0, 128, 255):   "Blue",
        (0, 255, 0):   "Green",  (0, 255, 128):   "Green",  (0, 255, 255):   "Blue",
        (128, 0, 0):   "Red",    (128, 0, 128):   "Purple", (128, 0, 255):   "Purple",
        (128, 128, 0): "Green",  (128, 128, 128): "Gray",   (128, 128, 255): "Purple",
        (128, 255, 0): "Green",  (128, 255, 128): "Green",  (128, 255, 255): "Blue",
        (255, 0, 0):   "Red",    (255, 0, 128):   "Pink",   (255, 0, 255):   "Purple",
        (255, 128, 0): "Orange", (255, 128, 128): "Orange", (255, 128, 255): "Pink",
        (255, 255, 0): "Yellow", (255, 255, 128): "Yellow", (255, 255, 255): "White"
    }

    # helper function to round values of 0-255 to the nearest of 0, 128, or 255
    def roundColorValue(value: int) -> int:
        if value < 85:
            return 0
        if value < 170:
            return 128
        return 255

    validString: bool = re.match(r"[0-9a-fA-F]{8}", hexString) is not None
    if validString:
        # for OpenCV type CV_8U (mode=24) the expected order of bytes is BGRA
        blue: int = roundColorValue(int(hexString[0:2], 16))
        green: int = roundColorValue(int(hexString[2:4], 16))
        red: int = roundColorValue(int(hexString[4:6], 16))
        alpha: int = int(hexString[6:8], 16)

        if alpha < 128:
            return "None"  # any pixel with less than 50% opacity is considered as color "None"
        return colors[(red, green, blue)]

    return "None"  # any input that is not in valid format is considered as color "None"
'''

# COMMAND ----------

# The annotations for the four images with the most colored non-transparent pixels
'''
mostColoredPixels: List[str] = ???

print("The annotations for the four images with the most colored non-transparent pixels:")
for image in mostColoredPixels:
    print(f"- {image}")
print("============================================================")


# The annotations for the five images having the lowest ratio of colored vs. transparent pixels
leastColoredPixels: List[str] = ???

print("The annotations for the five images having the lowest ratio of colored vs. transparent pixels:")
for image in leastColoredPixels:
    print(f"- {image}")
'''

# COMMAND ----------

# The three most common colors in the Finnish flag image:
'''
finnishFlagColors: List[str] = ???

# The percentages of the colored pixels for each common color in the Finnish flag image:
finnishColorShares: List[float] = ???

print("The colors and their percentage shares in the image for the Finnish flag:")
for color, share in zip(finnishFlagColors, finnishColorShares):
    print(f"- color: {color}, share: {share}")
print("============================================================")


# The number of images that have their most common three colors as, Blue-Yellow-Black, in that exact order:
blueYellowBlackCount: int = ???

print(f"The number of images that have, Blue-Yellow-Black, as the most common colors: {blueYellowBlackCount}")
'''

# COMMAND ----------

# The annotations for the five images with the most red pixels among the image group activities:
'''
redImageNames: List[str] = ???

# The number of red pixels in the five images with the most red pixels among the image group activities:
redPixelAmounts: List[int] = ???

print("The annotations and red pixel counts for the five images with the most red pixels among the image group 'activities':")
for color, pixel_count in zip(redImageNames, redPixelAmounts):
    print(f"- {color} (red pixels: {pixel_count})")
''' 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 4 - Machine learning tasks (2 points)
# MAGIC
# MAGIC This advanced task involves experimenting with the classifiers provided by the Spark machine learning library. Time series data collected in the [ProCem](https://www.senecc.fi/projects/procem-2) research project is used as the training and test data. Similar data in a slightly different format was used in the first tasks of weekly exercise 3.
# MAGIC
# MAGIC The folder `assignment/energy/procem_13m.parquet` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains the time series data in Parquet format.
# MAGIC
# MAGIC #### Data description
# MAGIC
# MAGIC The dataset contains time series data from a period of 13 months (from the beginning of May 2023 to the end of May 2024). Each row contains the average of the measured values for a single minute. The following columns are included in the data:
# MAGIC
# MAGIC | column name        | column type   | description |
# MAGIC | ------------------ | ------------- | ----------- |
# MAGIC | time               | long          | The UNIX timestamp in second precision |
# MAGIC | temperature        | double        | The temperature measured by the weather station on top of Sähkötalo (`°C`) |
# MAGIC | humidity           | double        | The humidity measured by the weather station on top of Sähkötalo (`%`) |
# MAGIC | wind_speed         | double        | The wind speed measured by the weather station on top of Sähkötalo (`m/s`) |
# MAGIC | power_tenants      | double        | The total combined electricity power used by the tenants on Kampusareena (`W`) |
# MAGIC | power_maintenance  | double        | The total combined electricity power used by the building maintenance systems on Kampusareena (`W`) |
# MAGIC | power_solar_panels | double        | The total electricity power produced by the solar panels on Kampusareena (`W`) |
# MAGIC | electricity_price  | double        | The market price for electricity in Finland (`€/MWh`) |
# MAGIC
# MAGIC There are some missing values that need to be removed before using the data for training or testing. However, only the minimal amount of rows should be removed for each test case.
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC - The main task is to train and test a machine learning model with [Random forest classifier](https://spark.apache.org/docs/3.5.0/ml-classification-regression.html#random-forests) in six different cases:
# MAGIC     - Predict the month (1-12) using the three weather measurements (temperature, humidity, and wind speed) as input
# MAGIC     - Predict the month (1-12) using the three power measurements (tenants, maintenance, and solar panels) as input
# MAGIC     - Predict the month (1-12) using all seven measurements (weather values, power values, and price) as input
# MAGIC     - Predict the hour of the day (0-23) using the three weather measurements (temperature, humidity, and wind speed) as input
# MAGIC     - Predict the hour of the day (0-23) using the three power measurements (tenants, maintenance, and solar panels) as input
# MAGIC     - Predict the hour of the day (0-23) using all seven measurements (weather values, power values, and price) as input
# MAGIC - For each of the six case you are asked to:
# MAGIC     1. Clean the source dataset from rows with missing values.
# MAGIC     2. Split the dataset into training and test parts.
# MAGIC     3. Train the ML model using a Random forest classifier with case-specific input and prediction.
# MAGIC     4. Evaluate the accuracy of the model with Spark built-in multiclass classification evaluator.
# MAGIC     5. Further evaluate the accuracy of the model with a custom build evaluator which should do the following:
# MAGIC         - calculate the percentage of correct predictions
# MAGIC             - this should correspond to the accuracy value from the built-in accuracy evaluator
# MAGIC         - calculate the percentage of predictions that were at most one away from the correct predictions taking into account the cyclic nature of the month and hour values:
# MAGIC             - if the correct month value was `5`, then acceptable predictions would be `4`, `5`, or `6`
# MAGIC             - if the correct month value was `1`, then acceptable predictions would be `12`, `1`, or `2`
# MAGIC             - if the correct month value was `12`, then acceptable predictions would be `11`, `12`, or `1`
# MAGIC         - calculate the percentage of predictions that were at most two away from the correct predictions taking into account the cyclic nature of the month and hour values:
# MAGIC             - if the correct month value was `5`, then acceptable predictions would be from `3` to `7`
# MAGIC             - if the correct month value was `1`, then acceptable predictions would be from `11` to `12` and from `1` to `3`
# MAGIC             - if the correct month value was `12`, then acceptable predictions would be from `10` to `12` and from `1` to `2`
# MAGIC         - calculate the average probability the model predicts for the correct value
# MAGIC             - the probabilities for a single prediction can be found from the `probability` column after the predictions have been made with the model
# MAGIC - As the final part of this advanced task, you are asked to do the same experiments (training+evaluation) with two further cases of your own choosing:
# MAGIC     - you can decide on the input columns yourself
# MAGIC     - you can decide the predicted attribute yourself
# MAGIC     - you can try some other classifier other than the random forest one if you want
# MAGIC
# MAGIC In all cases you are free to choose the training parameters as you wish.<br>
# MAGIC Note that it is advisable that while you are building your task code to only use a portion of the full 13-month dataset in the initial experiments.

# COMMAND ----------

# the structure of the code and the output format is left to the group's discretion
# the example output notebook can be used as inspiration

from pyspark.sql import functions as F

url = "abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/energy/procem_13m.parquet/procem.parquet"
procemDF: DataFrame = spark.read\
    .option("header", "true")\
    .option("sep", ",")\
    .option("inferSchema", "true")\
    .parquet(url)

procemDF.show()

# COMMAND ----------

procemDF = procemDF.dropna()
procemDF = procemDF.withColumn('time', F.from_utc_timestamp(F.from_unixtime(F.col('time')), 'Europe/Helsinki'))
monthDF = procemDF.withColumn('month', F.month(F.col('time')))
monthDF.show()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import RDD
from pyspark.sql.types import FloatType

class customEvaluator():
    def __init__(self, predictions, labelCol, predictionCol):
        self.labelCol = labelCol
        self.predictions = predictions
        self.predictionCol = predictionCol

    def percentage(self):
        correct = self.predictions.filter(F.col(self.predictionCol) == F.col(self.labelCol)).count()
        return round(100 * correct / self.predictions.count(), 2)

    def within_n_months(self, n=1):
        tmp = self.predictions.withColumn("diff", F.abs(F.col(self.labelCol) - F.col(self.predictionCol)))
        diff = 12-n if self.labelCol.lower() == "month" else 24-n
        correct = tmp.filter((tmp[self.labelCol] <= n) | (tmp[self.labelCol] == diff))
        return correct.count() / predictions.count()

    def avg_probability(self):
        extract_prob = F.udf(lambda prob, label: float(prob[label]), FloatType())
        probDF = self.predictions.withColumn("corrProb", extract_prob(F.col('probability'), F.col(self.labelCol)))
        avg_correct_prob = probDF.select(F.mean(F.col("corrProb"))).collect()[0][0]
        return avg_correct_prob

# COMMAND ----------

assembler = VectorAssembler(inputCols=["temperature", "humidity", "wind_speed"], outputCol="features")
month_threeweatherDF = assembler.transform(monthDF)

trainDF, testDF = month_threeweatherDF.randomSplit([0.8, 0.2])
rf = RandomForestClassifier(featuresCol="features", labelCol="month", probabilityCol="probability")
rf_model = rf.fit(trainDF)
predictions = rf_model.transform(testDF)

evaluator = MulticlassClassificationEvaluator(labelCol="month", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"MulticlassClassificationEvaluator Accuracy: {accuracy}")

custom_evaluator = customEvaluator(predictions=predictions, labelCol="month", predictionCol="prediction")
prc = custom_evaluator.percentage()
print(f"Custom Accuracy: {prc}")

within_one_month_percentage = custom_evaluator.within_n_months(n=1)
print(f"Percentage within one month: {within_one_month_percentage}")

within_two_months_percentage = custom_evaluator.within_n_months(n=2)
print(f"Percentage within two months: {within_two_months_percentage}")

avg_prob = custom_evaluator.avg_probability()
print(f"Average probability for correct predictions: {avg_prob}")


# COMMAND ----------

assembler = VectorAssembler(inputCols=["power_tenants", "power_maintenance", "power_solar_panels"], outputCol="features")
month_threepowerDF = assembler.transform(monthDF)

trainDF, testDF = month_threepowerDF.randomSplit([0.8, 0.2])
rf = RandomForestClassifier(featuresCol="features", labelCol="month")
rf_model = rf.fit(trainDF)
predictions = rf_model.transform(testDF)

evaluator = MulticlassClassificationEvaluator(labelCol="month", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"MulticlassClassificationEvaluator Accuracy: {accuracy}")

custom_evaluator = customEvaluator(predictions=predictions, labelCol="month", predictionCol="prediction")
prc = custom_evaluator.percentage()
print(f"Custom Accuracy: {prc}")

within_one_month_percentage = custom_evaluator.within_n_months(n=1)
print(f"Percentage within one month: {within_one_month_percentage}")

within_two_months_percentage = custom_evaluator.within_n_months(n=2)
print(f"Percentage within two months: {within_two_months_percentage}")

avg_prob = custom_evaluator.avg_probability()
print(f"Average probability for correct predictions: {avg_prob}")

# COMMAND ----------

assembler = VectorAssembler(inputCols=["temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price"], outputCol="features")
month_sevenfeaturesDF = assembler.transform(monthDF)

trainDF, testDF = month_sevenfeaturesDF.randomSplit([0.8, 0.2])
rf = RandomForestClassifier(featuresCol="features", labelCol="month")
rf_model = rf.fit(trainDF)
predictions = rf_model.transform(testDF)

evaluator = MulticlassClassificationEvaluator(labelCol="month", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"MulticlassClassificationEvaluator Accuracy: {accuracy}")

custom_evaluator = customEvaluator(predictions=predictions, labelCol="month", predictionCol="prediction")
prc = custom_evaluator.percentage()
print(f"Custom Accuracy: {prc}")

within_one_month_percentage = custom_evaluator.within_n_months(n=1)
print(f"Percentage within one month: {within_one_month_percentage}")

within_two_months_percentage = custom_evaluator.within_n_months(n=2)
print(f"Percentage within two months: {within_two_months_percentage}")

avg_prob = custom_evaluator.avg_probability()
print(f"Average probability for correct predictions: {avg_prob}")

# COMMAND ----------

hourDF = procemDF.withColumn('hour', F.hour(F.col('time')))
hourDF.show()

# COMMAND ----------

assembler = VectorAssembler(inputCols=["temperature", "humidity", "wind_speed"], outputCol="features")
hour_threeweatherDF = assembler.transform(hourDF)

trainDF, testDF = hour_threeweatherDF.randomSplit([0.8, 0.2])
rf = RandomForestClassifier(featuresCol="features", labelCol="hour")
rf_model = rf.fit(trainDF)
predictions = rf_model.transform(testDF)

evaluator = MulticlassClassificationEvaluator(labelCol="hour", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"MulticlassClassificationEvaluator Accuracy: {accuracy}")

custom_evaluator = customEvaluator(predictions=predictions, labelCol="hour", predictionCol="prediction")
prc = custom_evaluator.percentage()
print(f"Custom Accuracy: {prc}")

within_one_month_percentage = custom_evaluator.within_n_months(n=1)
print(f"Percentage within one month: {within_one_month_percentage}")

within_two_months_percentage = custom_evaluator.within_n_months(n=2)
print(f"Percentage within two months: {within_two_months_percentage}")

avg_prob = custom_evaluator.avg_probability()
print(f"Average probability for correct predictions: {avg_prob}")

# COMMAND ----------

assembler = VectorAssembler(inputCols=["power_tenants", "power_maintenance", "power_solar_panels"], outputCol="features")
hour_threepowerDF = assembler.transform(hourDF)

trainDF, testDF = hour_threepowerDF.randomSplit([0.8, 0.2])
rf = RandomForestClassifier(featuresCol="features", labelCol="hour")
rf_model = rf.fit(trainDF)
predictions = rf_model.transform(testDF)

evaluator = MulticlassClassificationEvaluator(labelCol="hour", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"MulticlassClassificationEvaluator Accuracy: {accuracy}")

custom_evaluator = customEvaluator(predictions=predictions, labelCol="hour", predictionCol="prediction")
prc = custom_evaluator.percentage()
print(f"Custom Accuracy: {prc}")

within_one_month_percentage = custom_evaluator.within_n_months(n=1)
print(f"Percentage within one month: {within_one_month_percentage}")

within_two_months_percentage = custom_evaluator.within_n_months(n=2)
print(f"Percentage within two months: {within_two_months_percentage}")

avg_prob = custom_evaluator.avg_probability()
print(f"Average probability for correct predictions: {avg_prob}")

# COMMAND ----------

assembler = VectorAssembler(inputCols=["temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price"], outputCol="features")
hour_sevenfeaturesDF = assembler.transform(hourDF)

trainDF, testDF = hour_sevenfeaturesDF.randomSplit([0.8, 0.2])
rf = RandomForestClassifier(featuresCol="features", labelCol="hour")
rf_model = rf.fit(trainDF)
predictions = rf_model.transform(testDF)

evaluator = MulticlassClassificationEvaluator(labelCol="hour", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"MulticlassClassificationEvaluator Accuracy: {accuracy}")

custom_evaluator = customEvaluator(predictions=predictions, labelCol="hour", predictionCol="prediction")
prc = custom_evaluator.percentage()
print(f"Custom Accuracy: {prc}")

within_one_month_percentage = custom_evaluator.within_n_months(n=1)
print(f"Percentage within one month: {within_one_month_percentage}")

within_two_months_percentage = custom_evaluator.within_n_months(n=2)
print(f"Percentage within two months: {within_two_months_percentage}")

avg_prob = custom_evaluator.avg_probability()
print(f"Average probability for correct predictions: {avg_prob}")

# COMMAND ----------

from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

assembler = VectorAssembler(inputCols=["temperature", "humidity", "month", "electricity_price"], outputCol="features")
custom_procemDF = assembler.transform(monthDF)
trainDF, testDF = custom_procemDF.randomSplit([0.8, 0.2])

glr = GeneralizedLinearRegression(featuresCol="features", labelCol="power_tenants")
glr_model = glr.fit(trainDF)
glr_pred = glr_model.transform(testDF)

evaluator = RegressionEvaluator(labelCol="power_tenants", predictionCol="prediction", metricName="mae")
glr_mse = evaluator.evaluate(glr_pred)
print(f"GeneralizedLinearRegression MSE: {glr_mse}")

# COMMAND ----------


/*Query Writing*/


/*Point Query*/

/*1) Finding a specific match details like overs, team, batsmen and runs scored using match_id.*/

SELECT MATCH_ID, INNINGS_ID, OVER_ID, BALL_ID, 
       TEAM_NAME AS BATTING_TEAM, 
       PLAYER_NAME AS BATSMAN, 
       BATSMAN_SCORED AS RUNS
FROM BALL_BY_BALL 
INNER JOIN MATCH USING (MATCH_ID)
INNER JOIN PLAYER_MATCH USING (MATCH_ID)
INNER JOIN PLAYER USING (PLAYER_ID)
INNER JOIN TEAM USING (TEAM_ID)
WHERE MATCH_ID = 501271  ;

/*2)Finding the top-scoring batsman for each season*/

SELECT DISTINCT PLAYER_ID, PLAYER_NAME, BATTING_HAND, COUNTRY, season_year
FROM BALL_BY_BALL
INNER JOIN MATCH USING(MATCH_ID)
INNER JOIN PLAYER_MATCH USING(MATCH_ID)
INNER JOIN PLAYER USING(PLAYER_ID)
INNER JOIN SEASON USING(SEASON_ID)
WHERE PLAYER_ID=ORANGE_CAP_ID
ORDER BY SEASON_YEAR;

/*Scan Query

1)Finding the Players with most man of the match awards*/



SELECT PLAYER_ID, PLAYER_NAME, COUNT(*) AS MAN_OF_THE_MATCH_COUNT
FROM MATCH
INNER JOIN PLAYER_MATCH USING (MATCH_ID)
INNER JOIN PLAYER USING (PLAYER_ID)
GROUP BY PLAYER_ID, PLAYER_NAME
HAVING COUNT(*) > 10
ORDER BY MAN_OF_THE_MATCH_COUNT DESC;



/*2) List all matches with total runs scored:*/

SELECT Match_Id, SUM(BATSMAN_Scored) AS Total_Runs
FROM Match 
INNER JOIN Ball_by_Ball USING(MATCH_ID)
GROUP BY Match_Id;


/*RANGE QUERIES

1) Finding matches which was won by 50 runs or 3 wickets */

SELECT MATCH_ID, MATCH_DATE, TEAM_NAME_ID, OPPONENT_TEAM_ID, WIN_TYPE, WON_BY
FROM MATCH
WHERE (WIN_TYPE = 'by runs' AND WON_BY < 50)
   OR (WIN_TYPE = 'by wickets' AND WON_BY < 3);


/*2) Find all players who have scored between 50 and 100 runs in a single-inning*/

SELECT Player_Name, Match_Id, SUM(BATSMAN_Scored) AS Runs_In_Inning
FROM BALL_BY_BALL
INNER JOIN MATCH USING(MATCH_ID)
INNER JOIN PLAYER_MATCH USING(MATCH_ID)
INNER JOIN PLAYER USING(PLAYER_ID)
GROUP BY Player_Name, Match_Id, Innings_Id
HAVING SUM(Batsman_Scored) BETWEEN 50 AND 100;


/*Combined Range and Scan Query*/



SELECT MATCH_ID, SUM(BATSMAN_SCORED) AS RUNS, COUNT(DISTINCT OVER_ID) AS OVERS_FACED
FROM BALL_BY_BALL
INNER JOIN MATCH USING (MATCH_ID)
INNER JOIN SEASON USING (SEASON_ID)
INNER JOIN PLAYER_MATCH USING (MATCH_ID)
INNER JOIN PLAYER USING (PLAYER_ID)
WHERE SEASON_YEAR = 2016 AND PLAYER_NAME = 'V Kohli' 
GROUP BY MATCH_ID;



/*Aggregation with Join Query*/

SELECT TEAM.TEAM_NAME, SUM(BATSMAN_SCORED) AS TOTAL_RUNS
FROM BALL_BY_BALL
JOIN MATCH ON BALL_BY_BALL.MATCH_ID = MATCH.MATCH_ID
JOIN SEASON ON MATCH.SEASON_ID = SEASON.SEASON_ID
JOIN TEAM ON BALL_BY_BALL.TEAM_BATTING_ID = TEAM.TEAM_ID
WHERE SEASON.SEASON_YEAR = 2015
GROUP BY TEAM.TEAM_NAME
ORDER BY TOTAL_RUNS DESC;


/*Performance Tuning

1)Indexing on Point Query*/

SELECT MATCH_ID, INNINGS_ID, OVER_ID, BALL_ID, 
       TEAM_NAME AS BATTING_TEAM, 
       PLAYER_NAME AS BATSMAN, 
       BATSMAN_SCORED AS RUNS
FROM BALL_BY_BALL 
INNER JOIN MATCH USING (MATCH_ID)
INNER JOIN PLAYER_MATCH USING (MATCH_ID)
INNER JOIN PLAYER USING (PLAYER_ID)
INNER JOIN TEAM USING (TEAM_ID)
WHERE MATCH_ID = 501271  ;


/*Impact:
Indexing on Match_Id column: It helps the where clause to filter matches based on match_id.
Indexing on Ball_By_Ball column: It helps the join operator to link match_id with other columns.
Indexing on Player column: It helps the join operator to link player_id with other columns.
We can see that in the Join Operator the cost reduced from 297 to 27 because of the indexing.*/

/*Indexing on Scan Query*/


/*Scan Query*/
SELECT PLAYER_ID, PLAYER_NAME, COUNT(*) AS MAN_OF_THE_MATCH_COUNT
FROM MATCH
INNER JOIN PLAYER_MATCH USING (MATCH_ID)
INNER JOIN PLAYER USING (PLAYER_ID)
GROUP BY PLAYER_ID, PLAYER_NAME
HAVING COUNT(*) > 10
ORDER BY MAN_OF_THE_MATCH_COUNT DESC;


/*Indexing Plan*/

ALTER TABLE PLAYER_MATCH
ADD CONSTRAINT PLAYER_ID PRIMARY KEY (MATCH_ID,PLAYER_ID);


/*We can see that in the Join Operator the cost reduced from 178 to 24 because of the indexing.*/


/*Indexing on Combined Range and Scan Query*/


/*Combined Range and Scan Query*/
SELECT MATCH_ID, SUM(BATSMAN_SCORED) AS RUNS, COUNT(DISTINCT OVER_ID) AS OVERS_FACED
FROM BALL_BY_BALL
INNER JOIN MATCH USING (MATCH_ID)
INNER JOIN SEASON USING (SEASON_ID)
INNER JOIN PLAYER_MATCH USING (MATCH_ID)
INNER JOIN PLAYER USING (PLAYER_ID)
WHERE SEASON_YEAR = 2016 AND PLAYER_NAME = 'V Kohli' 
GROUP BY MATCH_ID;



/*Indexing Plan*/
ALTER TABLE SEASON 
ADD CONSTRAINT SEASON_ID PRIMARY KEY (SEASON_ID);

ALTER TABLE TEAM
ADD CONSTRAINT TEAM_ID PRIMARY KEY (TEAM_ID);



/*We can see that in the Join Operator the cost reduced from 299 to 198 because of the indexing.*/




Data Visualisation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ball_by_ball_data=pd.read_csv("C:/Users/manvi/OneDrive/Desktop/Ball_by_Ball.csv")
match_data=pd.read_csv("C:/Users/manvi/OneDrive/Desktop/Match.csv")
player_data=pd.read_csv("C:/Users/manvi/OneDrive/Desktop/Player.csv")
playermatch_data=pd.read_csv("C:/Users/manvi/OneDrive/Desktop/Player_Match.csv")
season_data=pd.read_csv("C:/Users/manvi/OneDrive/Desktop/Season.csv")
Team_data=pd.read_csv("C:/Users/manvi/OneDrive/Desktop/Team.csv")

1)Extras bowled in the entire IPL

ball_by_ball_data['Extra_Type'].replace(' ', np.nan, inplace=True)
ball_by_ball_data['Extra_Type'].dropna()
plt.figure(figsize=(15,5))
sns.countplot(x='Extra_Type', data=ball_by_ball_data)
sns.set_context("talk")
plt.ylabel("No of Extras",fontsize = 20, weight = 'bold')
plt.xlabel("Types of extras",fontsize = 20, weight = 'bold')
plt.title("Extras bowled",fontsize = 20, weight = 'bold');
plt.show()




As you can see most of the Extras are Wides which are nearly 3500.
There are very less penalty runs.


2)Different types of Dismissals

ball_by_ball_data['Dissimal_Type'].replace(' ', np.nan, inplace=True)
ball_by_ball_data['Dissimal_Type'].dropna()
plt.figure(figsize=(25,5))
sns.countplot(x='Dissimal_Type', data=ball_by_ball_data)
sns.set_context("talk")
plt.ylabel("No of dismissals",fontsize = 20, weight = 'bold')
plt.xlabel("Types of dismissals",fontsize = 20, weight = 'bold')
plt.title("Total Dismissals",fontsize = 20, weight = 'bold');
plt.show()






3)Toss Decisions
plt.figure(figsize=(25,5))
sns.countplot(x='Toss_Decision', data=match_data)
sns.set_context("talk")
plt.ylabel("Toss Decisions",fontsize = 20, weight = 'bold')
plt.xlabel("Bat or Field",fontsize = 20, weight = 'bold')
plt.title("Total Dismissals",fontsize = 20, weight = 'bold');
plt.show()


4)Number of Players from different countries 

plt.figure(figsize=(12,6))
sns.countplot(x='Country', data=player_data)
sns.set_context('talk')
plt.xlabel("Country Names",fontsize=20,weight='bold')
plt.xticks( rotation=45, horizontalalignment='right')
plt.ylabel("Number of Players",fontsize=20,weight='bold')
plt.title("Total Number of Players that played IPL from each country",fontsize=25)
plt.show()





5)Host Countries of IPL

match_data['Host_Country'].unique()
         array(['India', 'South Africa', 'U.A.E'], dtype=object)

6)Number of matches played in each stadium

plt.figure(figsize=(20,14))
sns.countplot(y='Venue_Name', data=match_data)
plt.yticks(rotation='horizontal')
plt.xlabel("Number of Matches",fontsize = 25, weight = 'bold')
plt.ylabel("Stadium Name",fontsize = 25, weight = 'bold')
plt.title("Number of Matches played in each Stadium",fontsize = 30, weight = 'bold');
plt.show()










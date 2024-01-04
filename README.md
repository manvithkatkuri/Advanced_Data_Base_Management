The project entails creating a database system of Indian Premier League (IPL). It includes detailed records for each ball in every match, player performances, team details, 
and seasonal information. The "BALL_BY_BALL" entity captures specifics of each delivery, like which batsman faced it and the runs scored. "MATCH" details the match specifics, 
"PLAYER" contains details of each player, "PLAYER_MATCH" links players with specific matches, and "TEAM" and "SEASON" entities hold information about the IPL teams and the 
tournament years, respectively. This setup would be ideal for in-depth analysis of matches and player statistics over various IPL seasons that enables cricket fans as well 
as those in the sports industry such as journalists, bloggers, writers, etc, to be able to accurately and effectively lookup some of their favourite players and teams, the 
matches they were involved in, and other aspects on those leagues that they were involved in, such as the category under which it falls, winners, scores, etc.
Data Mining

Predicting the Result of the match based on team winning the Toss. We have considered the following columns "Toss_Decision","Match_Winner_Id","City_Name","Team_Name_Id","Opponent_Team_Id", for predicting of the result

To effectively balance overhead costs as well as retrieval times for reads from the database, appropriate indexing must be allocated to either a single or a combination of columns, 
while maintaining integrity and accuracy of the data retrieved.

1. Ball_by_Ball : Includes ball by ball details of all the 577 matches.
2. Match : Match metadata
3. Player : Player metadata
4. Player_Match : to know , who is the captain and keeper of the match , Includes every player who take part in match even If player haven't get a chance to either bat or bowl.
5. Season : Season wise details , Orange cap , Purple cap , Man_Of_The_Series
6. Team : Team Name

   
ER Diagram



![ER Diagram](https://github.com/manvithkatkuri/Advanced_Data_Base_Management/assets/102502757/8f803a6e-20f4-4985-8f1c-7208eebfc40e)



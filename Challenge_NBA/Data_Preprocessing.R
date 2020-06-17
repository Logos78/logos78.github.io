library(readr)
library(dplyr)
library(tidyverse)

training = read_csv(file = "train.csv")
results = read.csv(file = "train_results.csv", sep = ";")

# Récupération de certaines variables
ScoreMT = training[c(15831)]

ScoreHT = training[c(1,2,3,4)*3960-9]
colnames(ScoreHT) = c("Score HT1","Score HT2","Score HT3","Score HT4")
ScoreHT$"Score HT4" = ScoreHT$"Score HT4" - ScoreHT$"Score HT3"
ScoreHT$"Score HT3" = ScoreHT$"Score HT3" - ScoreHT$"Score HT2"
ScoreHT$"Score HT2" = ScoreHT$"Score HT2" - ScoreHT$"Score HT1"

### Découpage en 4 parties
training1 = training[c(1:3960)+1]
score1 = rowMeans(select(training1,starts_with("score")))
o_rebound1 = rowMeans(select(training1,starts_with("offensive rebound")))
d_rebound1 = rowMeans(select(training1,starts_with("defensive rebound")))
o_foul1 = rowMeans(select(training1,starts_with("offensive foul")))
d_foul1 = rowMeans(select(training1,starts_with("defensive foul")))
assist1 = rowMeans(select(training1,starts_with("assist")))
lost_ball1 = rowMeans(select(training1,starts_with("lost ball")))
steals1 = rowMeans(select(training1,starts_with("steals")))
bad_pass1 = rowMeans(select(training1,starts_with("bad pass")))
block1 = rowMeans(select(training1,starts_with("block")))
miss1 = rowMeans(select(training1,starts_with("miss")))
training1_final = cbind(score1,o_rebound1,d_rebound1,o_foul1,d_foul1,assist1,lost_ball1,steals1,bad_pass1,block1,miss1)
colname6 = c("Score 6", "Offensive rebound 6", "Defensive rebound 6", "Offensive foul 6", "Defensive foul 6", "Assist 6", "Lost ball 6", "Steals 6", "Bad pass 6", "Block 6", "Miss 6")

training2 = training[c(1:3960)+1+3960]
score2 = rowMeans(select(training2,starts_with("score")))
o_rebound2 = rowMeans(select(training2,starts_with("offensive rebound")))
d_rebound2 = rowMeans(select(training2,starts_with("defensive rebound")))
o_foul2 = rowMeans(select(training2,starts_with("offensive foul")))
d_foul2 = rowMeans(select(training2,starts_with("defensive foul")))
assist2 = rowMeans(select(training2,starts_with("assist")))
lost_ball2 = rowMeans(select(training2,starts_with("lost ball")))
steals2 = rowMeans(select(training2,starts_with("steals")))
bad_pass2 = rowMeans(select(training2,starts_with("bad pass")))
block2 = rowMeans(select(training2,starts_with("block")))
miss2 = rowMeans(select(training2,starts_with("miss")))
training2_final = cbind(score2,o_rebound2,d_rebound2,o_foul2,d_foul2,assist2,lost_ball2,steals2,bad_pass2,block2,miss2)
colname12 = c("Score 12", "Offensive rebound 12", "Defensive rebound 12", "Offensive foul 12", "Defensive foul 12", "Assist 12", "Lost ball 12", "Steals 12", "Bad pass 12", "Block 12", "Miss 12")

training3 = training[c(1:3960)+1+3960+3960]
score3 = rowMeans(select(training3,starts_with("score")))
o_rebound3 = rowMeans(select(training3,starts_with("offensive rebound")))
d_rebound3 = rowMeans(select(training3,starts_with("defensive rebound")))
o_foul3 = rowMeans(select(training3,starts_with("offensive foul")))
d_foul3 = rowMeans(select(training3,starts_with("defensive foul")))
assist3 = rowMeans(select(training3,starts_with("assist")))
lost_ball3 = rowMeans(select(training3,starts_with("lost ball")))
steals3 = rowMeans(select(training3,starts_with("steals")))
bad_pass3 = rowMeans(select(training3,starts_with("bad pass")))
block3 = rowMeans(select(training3,starts_with("block")))
miss3 = rowMeans(select(training3,starts_with("miss")))
training3_final = cbind(score3,o_rebound3,d_rebound3,o_foul3,d_foul3,assist3,lost_ball3,steals3,bad_pass3,block3,miss3)
colname18 = c("Score 18", "Offensive rebound 18", "Defensive rebound 18", "Offensive foul 18", "Defensive foul 18", "Assist 18", "Lost ball 18", "Steals 18", "Bad pass 18", "Block 18", "Miss 18")

training4 = training[c(1:3960)+1+3960+3960+3960]
score4 = rowMeans(select(training4,starts_with("score")))
o_rebound4 = rowMeans(select(training4,starts_with("offensive rebound")))
d_rebound4 = rowMeans(select(training4,starts_with("defensive rebound")))
o_foul4 = rowMeans(select(training4,starts_with("offensive foul")))
d_foul4 = rowMeans(select(training4,starts_with("defensive foul")))
assist4 = rowMeans(select(training4,starts_with("assist")))
lost_ball4 = rowMeans(select(training4,starts_with("lost ball")))
steals4 = rowMeans(select(training4,starts_with("steals")))
bad_pass4 = rowMeans(select(training4,starts_with("bad pass")))
block4 = rowMeans(select(training4,starts_with("block")))
miss4 = rowMeans(select(training4,starts_with("miss")))
training4_final = cbind(score4,o_rebound4,d_rebound4,o_foul4,d_foul4,assist4,lost_ball4,steals4,bad_pass4,block4,miss4)
colname24 = c("Score 24", "Offensive rebound 24", "Defensive rebound 24", "Offensive foul 24", "Defensive foul 24", "Assist 24", "Lost ball 24", "Steals 24", "Bad pass 24", "Block 24", "Miss 24")


training = cbind(training$ID,training1_final,training2_final,training3_final,training4_final,results$label)
training = as.data.frame(training)
colnames(training) = c("ID",colname6, colname12, colname18, colname24, "Match winner")


# Nettoyage des variables inutiles
rm(training1)
rm(training1_final)
rm(training2)
rm(training2_final)
rm(training3)
rm(training3_final)
rm(training4)
rm(training4_final)
rm(results)

rm(score1)
rm(o_rebound1)
rm(d_rebound1)
rm(o_foul1)
rm(d_foul1)
rm(assist1)
rm(lost_ball1)
rm(steals1)
rm(bad_pass1)
rm(block1)
rm(miss1)

rm(score2)
rm(o_rebound2)
rm(d_rebound2)
rm(o_foul2)
rm(d_foul2)
rm(assist2)
rm(lost_ball2)
rm(steals2)
rm(bad_pass2)
rm(block2)
rm(miss2)

rm(score3)
rm(o_rebound3)
rm(d_rebound3)
rm(o_foul3)
rm(d_foul3)
rm(assist3)
rm(lost_ball3)
rm(steals3)
rm(bad_pass3)
rm(block3)
rm(miss3)

rm(score4)
rm(o_rebound4)
rm(d_rebound4)
rm(o_foul4)
rm(d_foul4)
rm(assist4)
rm(lost_ball4)
rm(steals4)
rm(bad_pass4)
rm(block4)
rm(miss4)


# Retrait des variables Defensive foul
drop <- c("Defensive foul 6", "Defensive foul 12", "Defensive foul 18", "Defensive foul 24")
training = training[,!(names(training) %in% drop)]
rm(drop)

# Ajout de ScoreMT
colnames(ScoreMT) = c("ScoreMT")
training = cbind(training[1:41],ScoreHT,ScoreMT,training[42])

# Retrait des valeurs aberrantes
training = training[training$ID != 1939 & training$ID != 4058,]
ScoreMT = training[46]
ScoreHT = training[42:45]

# Création du fichier
write.table(training,file = "training.csv", sep = ",", row.names = FALSE)
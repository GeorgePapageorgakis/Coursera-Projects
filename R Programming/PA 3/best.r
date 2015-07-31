# takes two arguments: 
# @state: the 2-character abbreviated name of a state 
# @outcome: an outcome name. 
#
# Function reads the outcome-of-care-measures.csv file and returns a character 
# vector with the name of the hospital that has the best (i.e. lowest) 30-day 
# mortality for the specified outcome in that state. The hospital name is the 
# name provided in the Hospital.Name variable. The @outcomes can be one of 
# "heart attack", "heart failure", or "pneumonia". Hospitals that do not have 
# data on a particular outcome should be excluded from the set of hospitals 
# when deciding the rankings.

best <- function(state, outcome) {
	idx <- 0
	## Read outcome data
	setwd("D:/Dropbox/Educational/Coursera - EDx/R programming/PA 3")
	out_data <- read.csv("outcome-of-care-measures.csv", colClasses = "character")
	
	## Remove warnings about NAs being introduced
	out_data[,11] <- suppressWarnings(as.numeric(out_data[,11]))
	out_data[,17] <- suppressWarnings(as.numeric(out_data[,17]))
	out_data[,23] <- suppressWarnings(as.numeric(out_data[,23]))
	
	## Check that @state and @outcome are valid
	if ( !(state %in% unique(out_data$State)) ) 
		stop('invalid state') 
	if ( !(outcome %in% c('heart attack', 'heart failure', 'pneumonia')) )
		stop('invalid outcome')
	
	if 		(outcome == 'heart attack'	)	idx <- 11
    else if (outcome == 'heart failure'	)	idx <- 17
    else if (outcome == 'pneumonia'		)	idx <- 23
	
	#keep the subset of data with the given @state and remove NA rows in idx col
	out_data <- subset(out_data, out_data$State==state & !is.na(out_data[,idx]) & 
								out_data[,idx] != 'Not Available', select = c(2, idx))
	
	#return the subset rows that have the minimum value
	out_data <- subset(out_data, out_data[,2] == min(out_data[,2]))
	
	## Return hospital name in that state with lowest 30-day death rate
	out_data$Hospital.Name[order(out_data[,1])[1]]
}
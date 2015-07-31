rankhospital <- function(state, outcome, num = "best") {
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
	
	if 	(outcome == 'heart attack'	)	idx <- 11
    	else if (outcome == 'heart failure'	)	idx <- 17
    	else if (outcome == 'pneumonia'		)	idx <- 23
	
	#keep the subset of data with the given @state and remove NA rows in idx col
	out_data <- subset(out_data, out_data$State==state & !is.na(out_data[,idx]) & 
					out_data[,idx] != 'Not Available', select = c(2, idx))
	
	## Check that @num is valid
	if ( is.integer(num) & num > nrow(out_data))	#length(unique(out_data[,2]))
		return('NA')
	else if (num == 'best')
		num <- 1
	else if (num == 'worst')
		num <- nrow(out_data)
	
	## Return hospital name in that state with the given rank
	## 30-day death rate
	out_data[order(as.numeric(out_data[,2]),out_data[,1]), ] [num, 1]
}

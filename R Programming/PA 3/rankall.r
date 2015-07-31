rankall <- function(outcome, num = "best") {
	idx <- 0
	## Read @outcome data
	setwd("D:/Dropbox/Educational/Coursera - EDx/R programming/PA 3")
	out_data <- read.csv("outcome-of-care-measures.csv", colClasses = "character")
	
	## Remove warnings about NAs being introduced
	out_data[,11] <- suppressWarnings(as.numeric(out_data[,11]))
	out_data[,17] <- suppressWarnings(as.numeric(out_data[,17]))
	out_data[,23] <- suppressWarnings(as.numeric(out_data[,23]))
	
	## Check that @outcome are valid
	if ( !(outcome %in% c('heart attack', 'heart failure', 'pneumonia')) )
		stop('invalid outcome')
	
	if 	(outcome == 'heart attack'	)	idx <- 11
	else if (outcome == 'heart failure'	)	idx <- 17
    	else if (outcome == 'pneumonia'		)	idx <- 23
	
	#create an empty list for the data frame outcomes
    	hospitals <- c()
	
	#create a list of the state acronyms and sort them ascending
	states <- unique(out_data$State)
	states <- states[order(states)]
	
	## For each state, find the hospital of the given rank
	for (st in states) {
		sel <- 0
		#keep the subset of data with the @state and remove NA rows in idx col
		out <- subset(out_data, out_data$State==st & !is.na(out_data[,idx]) & 
					out_data[,idx] != 'Not Available', select = c(2, idx))
		## Check that @num is valid
		if ( is.integer(num) & num > nrow(out)){
			hospitals <- append(hospitals, 'NA')
			next
		}
		else if (num == 'best')
			sel <- 1
		else if (num == 'worst')
			sel <- nrow(out)
		
		## Append the hospital name in that state with the given rank
		## 30-day death rate in the hospitals list
		hospitals <- append(hospitals, out[order(as.numeric(out[,2]), out[,1]), ] [sel, 1])
	}
	
	## Return a data frame with the hospital names and the
	## (abbreviated) state name
    	data.frame(hospital = hospitals, state = states)
}

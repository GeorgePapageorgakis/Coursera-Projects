#set working directory for the project
setwd("D:/Dropbox/Educational/Coursera - EDx/R programming/PA 1")

#########################################################################
## Part 2

# Write a function that reads a directory full of files and reports the
# number of completely observed cases in each data file. The function 
# should return a data frame where the first column is the name of the 
# file and the second column is the number of complete cases. A prototype 
# of this function follows 

complete <- function(directory, id = 1:332) {
	## 'directory' is a character vector of length 1 indicating
	## the location of the CSV files

	## 'id' is an integer vector indicating the monitor ID numbers
	## to be used

	## Return a data frame of the form:
	## id nobs
	## 1  117
	## 2  1041
	## ...
	## where 'id' is the monitor ID number and 'nobs' is the
	## number of complete cases
	
	#create a list of all csv files in id range(tilde expansion)
	files <- list.files(path = directory, full.names = TRUE)[id]
	
	#returns the @sum of all the values present in the complete 
	#cases. @complete.cases returns a logical vector indicating 
	#which cases have no missing values (are complete). 
	count_comp_file <- function(fname){
		sum(complete.cases(read.csv(file = fname)))
	}
	
	#return the data frame, @lapply returns a list of the same length
	#as files, each element of which is the result of applying 
	#"count_comp_file" to the corresponding element of files
	#@unlist simplifies list structure to produce a vector which 
	#contains all the atomic components which occur in the list. 
	data.frame(id = id, nobs = unlist(lapply(files, count_comp_file)))
}

##   id nobs
## 1  1  117
#complete("specdata", 1)

##   id nobs
## 1 30  932
## 2 29  711
## 3 28  475
## 4 27  338
## 5 26  586
## 6 25  463
#complete("specdata", 30:25)

##   id nobs
## 1  3  243
#complete("specdata", 3)
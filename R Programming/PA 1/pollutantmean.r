#set working directory for the project
setwd("D:/Dropbox/Educational/Coursera - EDx/R programming/PA 1")

#Point and download zip
url <- "https://d396qusza40orc.cloudfront.net/rprog%2Fdata%2Fspecdata.zip"
destfile <- "rprog_data_specdata.zip"
download.file(url, destfile)

#unzip file
unzip("rprog_data_specdata.zip")

#set the directory path of data
#directory <- "specdata/"

#check files
#list.files("specdata")

###########################################################################
## Part 1

# Write a function named 'pollutantmean' that calculates the mean of a 
# pollutant (sulfate or nitrate) across a specified list of monitors. 
# The function 'pollutantmean' takes three arguments: 'directory', 
# 'pollutant', and 'id'. Given a vector monitor ID numbers, 'pollutantmean' 
# reads that monitors' particulate matter data from the directory specified 
# in the 'directory' argument and returns the mean of the pollutant across 
# all of the monitors, ignoring any missing values coded as NA.

pollutantmean <- function(directory, pollutant, id = 1:332) {
	## 'directory' is a character vector of length 1 indicating
	## the location of the CSV files

	## 'pollutant' is a character vector of length 1 indicating
	## the name of the pollutant for which we will calculate the
	## mean; either "sulfate" or "nitrate".
	
	## 'id' is an integer vector indicating the monitor ID numbers
	## to be used

	## Return the mean of the pollutant across all monitors list
	## in the 'id' vector (ignoring NA values)
	## NOTE: Do not round the result!
	
	#create an empty data frame
	pol_dframe <- data.frame()
	#create a list of all the csv files (tilde expansion of pathname)
	files <- list.files(path = directory, full.names = TRUE)
	
	#rbind all .csv files in the data frame
	for (i in id){
		pol_dframe <- rbind(pol_dframe, read.csv(files[i]))
	}
	mean(pol_dframe[, pollutant], na.rm = TRUE)
}

pollutantmean("specdata", "sulfate", 1:10)	#[1] 4.064

pollutantmean("specdata", "nitrate", 70:72)	#[1] 1.706

pollutantmean("specdata", "nitrate", 23)	#[1] 1.281

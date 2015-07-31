#set working directory for the project
setwd("D:/Dropbox/Educational/Coursera - EDx/R programming/PA 1")

#######################################################################
##Part 3

#Write a function that takes a directory of data files and a threshold 
#for complete cases and calculates the correlation between sulfate and 
#nitrate for monitor locations where the number of completely observed 
#cases (on all variables) is greater than the threshold. The function 
#should return a vector of correlations for the monitors that meet the 
#threshold requirement. If no monitors meet the threshold requirement, 
#then the function should return a numeric vector of length 0.

corr <- function(directory, threshold = 0) {
	## 'directory' is a character vector of length 1 indicating
	## the location of the CSV files

	## 'threshold' is a numeric vector of length 1 indicating the
	## number of completely observed observations (on all
	## variables) required to compute the correlation between
	## nitrate and sulfate; the default is 0

	## Return a numeric vector of correlations
	## NOTE: Do not round the result!
	
	#create a list of all csv files
	files <- list.files(path = directory, full.names = TRUE)
	
	#create a data frame of the complete items in each file
	comp_df <- complete(directory)
	
	#create an empty vector
	corr_vect <- c()
	
	for (idx in 1:nrow(comp_df)){
		if (comp_df[idx, 2] > threshold){
			#read each file that is greater than the threshold 
			buff_file <- read.csv(files[idx])
			
			#remove NA values from columns
			buff_file <- buff_file[complete.cases(buff_file),]

			#get common columns of sulfate and nitrate without na values
			sulfate <- na.omit(buff_file[2])
			nitrate <- na.omit(buff_file[3])
			
			#append vector with the new value
			corr_vect <- c(corr_vect, cor(sulfate,nitrate))
		}
	}
	#return the vector
	corr_vect
}

#source("complete.r")
#source("corr.r")

cr <- corr("specdata", 150)
head(cr)
## [1] -0.01896 -0.14051 -0.04390 -0.06816 -0.12351 -0.07589

summary(cr)
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## -0.2110 -0.0500  0.0946  0.1250  0.2680  0.7630

cr <- corr("specdata", 400)
head(cr)
## [1] -0.01896 -0.04390 -0.06816 -0.07589  0.76313 -0.15783

summary(cr)
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## -0.1760 -0.0311  0.1000  0.1400  0.2680  0.7630

cr <- corr("specdata", 5000)
summary(cr)
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## 

length(cr)
## [1] 0

cr <- corr("specdata")
summary(cr)
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## -1.0000 -0.0528  0.1070  0.1370  0.2780  1.0000

length(cr)
## [1] 323



#This seems to refer to a conditional filter whereby the correlation is calculated 
#only for datasets which have fewer than a certain number (threshold) of missing 
#values (ie if there are fewer than x complete observations, the monitor is omitted). 
#Makes sense for very large datasets to make sure you don't get correlations if 
#there are unreasonably few observations. 
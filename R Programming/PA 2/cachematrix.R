# Creates a special "matrix", which is really a list containing a function to
#    set the value of the matrix
#    get the value of the matrix
#    set the value of the inverse matrix
#    get the value of the inverse matrix
makeCacheMatrix <- function(x = matrix()) {
	m <- NULL
	set <- function(y) {
	##caches the input matrix so that cacheSolve can check whether it has changed 
		x <<- y
		m <<- NULL
	}
	get <- function() x
	setinverse <- function(inverse) 
		m <<- inverse
	getinverse <- function() 
		m
	
	#creates a list to store the four functions
	list(set = set, get = get,
		setinverse = setinverse,
		getinverse = getinverse)
}

# The following function calculates the inverse of the "matrix" created
# with the above function. However, it first checks to see if the inverse has 
# already been calculated. If so, it gets the inverse from the cache and skips
# the computation. Otherwise, it calculates the inverse of the data and sets
# the matrix of the inverse in the cache via the setinverse() function.
cacheSolve <- function(x, ...) {
	# query if an inverse matrix has already been calculated
	m <- x$getinverse()
	# if there is a cached m matrix it will not be NULL
	if(!is.null(m)) {
		message("getting cached data")
		return(m)
	}
	# else if there is not a cached m matrix get the matrix to inverse it
	data <- x$get()
	#calculate the inverse matrix with solve()
	m <- solve(data, ...)
	# run setinverse() on the inverse to cache the inverse matrix
	x$setinverse(m)
	m
}

#x <- matrix( nrow=3, ncol=3, c(2,3,2, 1,2,1, 1,1,2))
#m = makeCacheMatrix(x)
#m$get()
#     [,1] [,2] [,3]
#[1,]    2    1    1
#[2,]    3    2    1
#[3,]    2    1    2

##first run not cached
#cacheSolve(m)
#     [,1] [,2] [,3]
#[1,]    3   -1   -1
#[2,]   -4    2    1
#[3,]   -1    0    1
# 
##second run, cached
#cacheSolve(m)
#getting cached data
#     [,1] [,2] [,3]
#[1,]    3   -1   -1
#[2,]   -4    2    1
#[3,]   -1    0    1
"""
Recursive class definition for a non-empty list of nodes
"""
total = 0
nonecount = 0
existcount = 0

class NodeList:
    """
    Basic class definition for non-empty lists using recursion
    """
    
    def __init__(self, val):
        """
        Create a list with one node
        """
        self._value = val
        self._next = None
     
    
    def append(self, val):
        """
        Append a node to an existing list of nodes
        """
        global total, nonecount, existcount
        total += 1
        #print "total evaluations", total
        print "Append ()"
        if self._next == None:
            nonecount += 1
            print "None branch"
            #print "None evaluations", nonecount
            new_node = NodeList(val)
            self._next = new_node
        else:
            existcount += 1
            print "append branch"
            #print "existcount evaluations", existcount
            self._next.append(val)
            

    def __str__(self):
        """
        Build standard string representation for list
        """
        if self._next == None:
            return "[" + str(self._value) + "]"
        else:
            rest_str = str(self._next)
            rest_str = rest_str[1 :]
            return "[" + str(self._value) + ", " + rest_str
    
def run_example():
    """
    Create some examples
    """
    global total
    print "Adding 1 \n"
    node_list = NodeList(1)
    print "total calls", total
    total = 0
    print "################\n"
    print "Adding 2 \n"
    node_list.append(2)
    print "total calls", total
    total = 0
    print "################\n"
    print "Adding 3 \n"
    node_list.append(3)
    print "total calls", total
    total = 0
    print "################\n"
    print "Adding 4 \n"
    node_list.append(4)
    print "total calls", total
    total = 0
    print "################\n"
    print "Adding 5 \n"
    node_list.append(5)
    print "total calls", total
    total = 0
    print "################\n"
    print "Adding 6 \n"
    node_list.append(6)
    print "total calls", total
    total = 0
    print "################\n"
    
    node_list.append(4)
    print node_list
    
    sub_list = NodeList(5)
    sub_list.append(6)
    
    print "################\n"
    node_list.append(sub_list)
    print node_list
    
run_example()
print "################"
print "evaluations", total, nonecount, existcount        
         
        
            
        
        
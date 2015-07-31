"""
Simulator for resource generation with upgrades
"""
import simpleplot
import math
import codeskulptor
codeskulptor.set_timeout(20)

# Plot options
STANDARD = True
LOGLOG = False

def resources_vs_time(upgrade_cost_increment, num_upgrade, plot_type):
    """
    Build function that performs unit upgrades with specified cost increments
    """
    current_time = 0.0
    total_resources_generated = 0.0
    result = []

    for idx in range( num_upgrade ):
        #cost for subsequent upgrade = initial upgrade + fixed increment
        new_rate = 1 + idx
        
        new_cost = 1 + idx  * upgrade_cost_increment
        
        total_resources_generated +=  new_cost
        current_time += new_cost/new_rate
        
        if plot_type:   
            result.append([current_time, total_resources_generated])       
        else:
            result.append([math.log(current_time), math.log(total_resources_generated)])
          
    return result


def test():
    """
    Testing code for resources_vs_time
    """
    data1 = resources_vs_time(0.5, 20, STANDARD)
    data2 = resources_vs_time(1.5, 10, STANDARD)
    #print data1
    #print data2
    
    #test1 = [[1.0, 1], [1.75, 2.5], [2.41666666667, 4.5], [3.04166666667, 7.0], [3.64166666667, 10.0], [4.225, 13.5], [4.79642857143, 17.5], [5.35892857143, 22.0], [5.91448412698, 27.0], [6.46448412698, 32.5], [7.00993867244, 38.5], [7.55160533911, 45.0], [8.09006687757, 52.0], [8.62578116328, 59.5], [9.15911449661, 67.5], [9.69036449661, 76.0], [10.2197762613, 85.0], [10.7475540391, 94.5], [11.2738698286, 104.5], [11.7988698286, 115.0]]
    #test2 = [[1.0, 1], [2.25, 3.5], [3.58333333333, 7.5], [4.95833333333, 13.0], [6.35833333333, 20.0], [7.775, 28.5], [9.20357142857, 38.5], [10.6410714286, 50.0], [12.085515873, 63.0], [13.535515873, 77.5]]
    
    print "\n###############  Question 1  ##################"
    question1_a = resources_vs_time(0.0, 10, STANDARD)
    question1_b = resources_vs_time(1.0, 10, STANDARD)
    #print question1_a
    print question1_b
    #simpleplot.plot_lines("Growth", 600, 600, "time", "total resources", [data1, data2])
    #simpleplot.plot_lines("Growth", 600, 600, "time", "total resources", [test1, test2])

    
    print "\n###############  Question 2  ##################"
    question2_a = resources_vs_time(0.0, 100, STANDARD)
    question2_b = resources_vs_time(0.5, 18, STANDARD)
    question2_c = resources_vs_time(1.0, 14, STANDARD)
    question2_d = resources_vs_time(2.0, 10, STANDARD)
    #print question2_a
    #print question2_b
    #print question2_c
    #print question2_d
    simpleplot.plot_lines("Growth", 600, 600, "time", "total resources", [question2_a, question2_b, question2_c, question2_d])
    
    
    print "\n###############  Question 3  ##################"
    question3 = resources_vs_time(0.0, 10, LOGLOG)
    #print question3
    #simpleplot.plot_lines("Growth", 600, 600, "time", "total resources", [question3])
    
    
    print "\n###############  Question 7  ##################"
    question7 = resources_vs_time(1.0, 10, LOGLOG)
    print question7
    #simpleplot.plot_lines("Growth", 600, 600, "time", "total resources", [question1_b])
      
    print "\n###############  Question 9  ##################"
    question9a = resources_vs_time(1.0, 10, STANDARD)
    question9b = resources_vs_time(1.0, 20, STANDARD)
    print question9a
    print question9b
   # simpleplot.plot_lines("Growth", 600, 600, "time", "total resources", [question1_b])
      

test()


# Sample output from the print statements for data1 and data2
#[[1.0, 1], [1.75, 2.5], [2.41666666667, 4.5], [3.04166666667, 7.0], [3.64166666667, 10.0], [4.225, 13.5], [4.79642857143, 17.5], [5.35892857143, 22.0], [5.91448412698, 27.0], [6.46448412698, 32.5], [7.00993867244, 38.5], [7.55160533911, 45.0], [8.09006687757, 52.0], [8.62578116328, 59.5], [9.15911449661, 67.5], [9.69036449661, 76.0], [10.2197762613, 85.0], [10.7475540391, 94.5], [11.2738698286, 104.5], [11.7988698286, 115.0]]
#[[1.0, 1], [2.25, 3.5], [3.58333333333, 7.5], [4.95833333333, 13.0], [6.35833333333, 20.0], [7.775, 28.5], [9.20357142857, 38.5], [10.6410714286, 50.0], [12.085515873, 63.0], [13.535515873, 77.5]]

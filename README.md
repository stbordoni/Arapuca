# Arapuca
Xe doping analysis with Arapuca detector in pDUNE

The main function is in doRoutine.py  
The other two .py files contains the functions which are called by the main

The main function accepts two arguments: 'first' (-f or --first) and 'last'  (-l or --last) 
Those two arguments are thought to split the run over the runlist. 
This is helpful to run in parallel over different runs or if new runs are added to the list just run over these new ones. 



N.B. The notebooks (.ipynb) are usually a bunch of lines of codes I use to test the different parts before implementing them in the doRoutine. Please do not pay attention to them


To run the code do:

python doRoutine.py -f 5  -l 10

in this case the execution is done over the 5th and the 9th element of the runlist

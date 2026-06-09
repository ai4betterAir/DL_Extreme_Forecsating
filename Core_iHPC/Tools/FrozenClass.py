class FrozenClass(object):
    __isfrozen = False
    def __setattr__(self, key, value):
#        if self.__isfrozen and not hasattr(self, key):
        if self.__isfrozen and key not in dir(self):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True
class Test(FrozenClass):
    def __init__(self):
        self.x = 42#
        self.y = 2**3

        self._freeze() # no new attributes after this point.

def run_main():
    '''
    https://stackoverflow.com/questions/3603502/prevent-creating-new-attributes-outside-init
    	
Late comment: I was using this recipe successfully for some time, until I changed an attribute to a property, where the getter was raising a NotImplementedError. It took me a long time to find out that this was due to the fact that hasattr actuall calls getattr, trows away the result and returns False in case of errors, see this blog. Found a workaround by replacing not hasattr(self, key) by key not in dir(self). This might be slower, but solved the problem for me
    '''
    a,b = Test(), Test()
    a.x = 10
    b.z = 10 # fails
    
######################################################    
if __name__ == '__main__':
    run_main()        

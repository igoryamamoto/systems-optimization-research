from numpy import array
import matplotlib.pyplot as plt
import numpy as np

class DMI :
    def __init__(self, path):
        f = open(path,'r');
        self.cabecalho = f.readline()
        f.readline() #	zero
        self.numInd = int(f.readline())
        self.numDep = int(f.readline())
        self.nCoef = int(f.readline())
        f.readline() #	zero
        self.tss = float(f.readline())
    	
        for i in range(self.numInd + self.numDep + 1) :
            f.readline() #	zero
        
        self.rd = {}
        for dep in range(self.numDep) :
            tagDep = f.readline().split()[0];
            self.rd[tagDep] = {}
            for i in range(10) :
                f.readline() #	zero
            
            for ind in range(self.numInd) :
                tagInd = f.readline().split()[0];
                self.rd[tagDep][tagInd] = []
                for i in range((self.nCoef+4)//5):
                    self.rd[tagDep][tagInd] += map(float,f.readline().split())
                self.rd[tagDep][tagInd] = array(self.rd[tagDep][tagInd]);
        self.tagDep = list(self.rd.keys())
        self.tagInd = list(self.rd[self.tagDep[0]].keys())

    def save(self, path) :
        f = open(path, 'w')
        f.write(self.cabecalho) 
        f.write('0\n') #	zero
        f.write(str(self.numInd) + '\n')
        f.write(str(self.numDep) + '\n')
        f.write(str(self.nCoef) + '\n')
        f.write('0\n') #	zero
        f.write(str(self.tss) + '\n')
        
        for i in range(self.numInd + self.numDep + 1) :
            f.write('0\n')
        
        for dep in self.rd :
            f.write(dep.ljust(16) + '\n');
            for i in range(10) :
                f.write('0\n')
            for ind in self.rd[dep] :
               f.write(ind.ljust(16) + '\n');
               for i in range(self.nCoef):
                   f.write('    {:.11f}'.format(self.rd[dep][ind][i]))
                   if i%5 == 4:
                       f.write('\n')
    
    def plot(self):
        f, axss = plt.subplots(self.numDep,self.numInd)
        for i,dep in enumerate(self.rd):
            for j,ind in enumerate(self.rd[dep]):
                axss[i,j].plot(self.rd[dep][ind])
                
    def getArray(self):   
        return np.array([[self.rd[dep][indep] for indep in self.tagInd] for dep in self.tagDep])
    
if __name__ == '__main__':
    dmi = DMI('Modelo_Desbutanizadora_1.dmi')
    print('Dependentes:',dmi.tagDep)
    print('Independentes:',dmi.tagInd)
    print('Num. Coeficientes:',dmi.nCoef)
    print('Tamanho do array de dados',dmi.getArray().shape)
    dmi.plot()
    
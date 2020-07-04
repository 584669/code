import numpy as np
import matplotlib.pyplot as plt
class Mat():
    def __init__(self,min,max,u=1):
        self.min=min
        self.max = max
        self.u=u
    def mat_arctan_loss(self):
        x = np.array([i * 0.01 for i in range(-self.min, self.max)])
        y = np.where(np.abs(x) <= 1, 0.5 * x ** 2, np.abs(x) - 0.5)
        plt.plot(x, y, label="smoothl1")
        for i in range(-20,0,4):
            b=i*0.1
            b=round(b,2)
            a=round((1+0.5*b)/(np.arctan(1)+0.5),2)
            c=-self.u+a*np.arctan(1)-(np.log(2)/2)*b
            y= np.where(np.abs(x)<=1,a*np.abs(x)*np.arctan(np.abs(x))-(b/2)*np.log(1+x**2),self.u*np.abs(x)+c)
            plt.plot(x, y, label="b={},a={}".format(i, a))
        plt.title('loss_arctan')
        plt.legend(loc='upper left')
        plt.show()
    def mat_arctan_grad(self):
        x = np.array([i * 0.01 for i in range(-self.min, self.max)])
        y = np.where(np.abs(x) <= 1, x, np.where(x > 1, 1, -1))
        plt.plot(x, y, label="smoothl1")
        for i in range(-20, 0, 4):
            b = i * 0.1
            b = round(b, 3)
            a = round((self.u + 0.5 * b) / (np.arctan(1) + 0.5), 3)
            y_ = np.where(np.abs(x) <= 1, a * np.arctan(x) + (a - b) * (x/ (1 + x ** 2)),
                          np.where(x>1,self.u,-self.u))
            plt.plot(x, y_, label="b={},a={}".format(b, a), )
        plt.title('grad_arctan')
        plt.legend(loc='upper left')
        plt.show()
    def mat_ex_grad(self):
        x = np.array([i * 0.01 for i in range(-self.min, self.max)])
        y = np.where(np.abs(x) <= 1, x, np.where(x>1,1,-1))
        plt.plot(x, y, label="smoothl1")
        for i in range(1, 10,2):
            a = round(self.u / (1 - np.exp(-i)), 2)
            y_ = np.where(np.abs(x)<= 1, np.where(x>0,a * (1 - np.exp(-i * np.abs(x))),a * (-1 + np.exp(-i * np.abs(x)))),
                          np.where(x>1,self.u,-self.u))
            plt.plot(x, y_)
            plt.plot(x, y_, label="β={}".format(i), )
        plt.legend(loc='upper left')
        plt.title("grad_ex")
        plt.show()
    def mat_ex_loss(self):
        x=np.array([i*0.01 for i in range(-self.min, self.max)])
        y= np.where(np.abs(x)<=1,0.5*x**2,np.abs(x)-0.5)
        plt.plot(x, y, label="smoothl1")
        for b in range(1,10,2):
            a=round(self.u/(1-np.exp(-b)),2)
            c=a-self.u-(self.u/b)
            y_=np.where(np.abs(x)<=1,a*(np.abs(x)+(1/b)*(np.exp(-b*np.abs(x))-1)),self.u*np.abs(x)+c)
            plt.plot(x, y_, label="β={}".format(b))
        plt.legend(loc = 'upper left')
        plt.title("loss_ex")
        plt.show()
mat=Mat(0,130,1)
# mat.mat_arctan_loss()
# mat.mat_arctan_grad()
mat.mat_ex_loss()
mat.mat_ex_grad()


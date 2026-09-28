import numpy as np

def  grad(x):
    return 2*x

def cost(x):
    return x**2 - 2

def myGD(x0, eta):
    x=[x0]

    for it in range (100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return x, it

x0_input = float(input("Nhập điểm khởi tạo x0: "))    
eta_input = float(input("Nhập tốc độ học eta: "))

(x, e) = myGD(x0_input, eta_input)

print('Solution x = %f, cost = %f, after %d iterations' % (x[-1], cost(x[-1]), e))

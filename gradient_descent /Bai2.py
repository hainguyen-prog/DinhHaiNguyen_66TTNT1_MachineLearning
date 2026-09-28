import numpy as np

def cost(x):
    return 1/3*x**3 - x

def grad(x):
    return x**2 - 1

def myGD(x0, e):
    x = [x0]

    for it in range (100):
        x_new = x[-1] - e*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

x0_input = float(input("Nhập điểm khởi tạo x0: "))
eta_input = float(input("Nhập tốc độ học: "))

(x, e) = myGD(x0_input, eta_input)

print('Solution x = %f, cost = %f, after %d interations' (x[-1], cost(x[-1]), e))

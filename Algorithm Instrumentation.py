import numpy as np
import sympy as sym



x= sym.Symbol("x")
y = sym.Symbol("y")
z = sym.Symbol("z")
w=sym.Symbol("w")

def newton (f,df,x0,epsilon,max_iter):
  xn = x0
  for i in range (0,max_iter):
    fxn =f.subs(w,x0)
    if fxn ==0:
      return xn
    Dfxn = df.subs(w,x0)
    if Dfxn == 0:
      print('La derivada es 0. No se encontró solución.')
      return None
    xn = xn - fxn/Dfxn
    
    while abs(xn - x0)> epsilon:
        x0=xn
  return xn
    

def gradient_descent_v2( Fx ,x_inicial, Max_Iteraciones=100, tolerance=0.00001):
    
    x_k  = x_inicial
    
    
    
    i=0
    
    
    for i in range(Max_Iteraciones):
        
       
        
        sk_x= -sym.diff(Fx,x,1).subs(x,x_k[0])
        sk_y= -sym.diff(Fx,y,1).subs(y,x_k[1])
        sk_z= -sym.diff(Fx,z,1).subs(z,x_k[2])
        sk_0 =sym.Matrix([[sk_x],[sk_y],[sk_z]])

        
        Fw= x_k+ sk_0*w
       
        
        f_lax = Fx.subs(x,Fw[0])
        f_lax = f_lax.subs(y,Fw[1])
        f_lamb = f_lax.subs(z,Fw[2])
       
 
        Fobj = sym.diff(f_lamb,w,1)

        dFobj= sym.diff(f_lamb,w,2)

        y_k = newton (Fobj, dFobj,0.05 , tolerance,300)
         
        delta = y_k*sk_0
        
        x_k1 = x_k + delta
          
        i+=1
        x_k = x_k1
        
        if np.all(np.abs(delta) <= tolerance):
            break
        
        
       
    return x_k , i 
            
           
        


f = (10*x**2) + (2*y**2) + (40*z**2)

pi =sym.Matrix([-10,-40,2])


print(gradient_descent_v2(f,pi ,Max_Iteraciones=150,tolerance= 0.001))




"""
Se decidio optar por sk =-Grad(f) ya que al actualizar lambda de forma iterativa
podemos ajustar la longitud de paso convenientemente minimizando f(x_k- lambda*Grad(f))
evitando de cierta forma la oscilacion de la direccion.

"""






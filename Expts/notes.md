
#AR
u_n+1 = NO(u_n)

#Euler
u_n+1 = u_n + dt*NO(u)

#PRE 
PRE(u) = D_t(u) - D_xx(u)

#Physics-Informed Supervisory Loss #PISL - implicit as well. 
loss = MSE(PRE(u_pred)- PRE(u_actual))
load -ascii Seite_145_Kapitel_10_2_data_t_y.csv;  
t = Seite_145_Kapitel_10_2_data_t_y(:,1);  
y = Seite_145_Kapitel_10_2_data_t_y(:,2:3);  
u = zeros(1,length(y));

Ts = 0.1;

R  = [20 0.0; 
       0 0.2];
			
Q1  = 100/9;
Q2  = 0.04/1000; 

Ad = [1  Ts  0.5*Ts^2  0;
      0  1       Ts    0;
      0  0       1     0; 
      0  0       0     1]; 
			
Bd = [0; 
      0;
      0;
      0;];

C  = [1 0 0  0; 
      0 0 1 -1];
			
GQG = [Q1*Ts^4/4 Q1*Ts^3/2 Q1*Ts^2/2 0;
       Q1*Ts^3/2 Q1*Ts^2   Q1*Ts     0;
       Q1*Ts^2/2 Q1*Ts     Q1        0;
            0      0       0         Q2];


%%% INITIALISIERUNG KALMAN-FILTER %%%
x = [y(1,1); 
     0.0; 
     y(1,2);
     0.0;];

P = 3*[1 0 0 0; 
       0 1 0 0; 
       0 0 1 0; 
       0 0 0 .01]; 

xStart = x
newState = Ad*x + Bd*u(k)
newCovariance = Ad*P*Ad' + GQG

%%%  ZYKLISCHE BERECHNUNG KALMAN-FILTER %%%
for k=1:3
  K = P*C'*pinv(C*P*C' + R);
  x = x + K*(y(k,:)' - C*x);
  P = (eye(length(Bd)) - K*C)*P;
  
  xUpdate = x
  pUpdate = P
  
  s(k)=x(1);  v(k)=x(2);  a(k)=x(3);  ao(k)=x(4);
  
  result = C * x 
 
  x = Ad*x + Bd*u(k);
  P = Ad*P*Ad' + GQG;
  
  xApply = x
  pApply = P
  
  
end


figure(1); 
subplot(411);
plot(t,ao,'r--'); grid on;
xlabel('Zeit[s]'); ylabel(' Offsets des Beschleunigungssignals [m/s^2]'); 
legend('geschätztes Signal');

subplot(412);
plot(t,y(:,2),'b-',t,a,'r--'); grid on;
xlabel('Zeit[s]'); ylabel('Beschleunigung [m/s^2]'); 
legend('gemessenes Signal','gefiltertes Signal');

subplot(413);
plot(t,v,'r--'); grid on;
xlabel('Zeit[s]'); ylabel('Geschwindigkeit [m/s]'); 
legend('geschätztes Signal');

subplot(414);
plot(t,y(:,1),'b-',t,s,'r--'); grid on;
xlabel('Zeit[s]'); ylabel('Position [m]'); 
legend('gemessenes Signal','gefiltertes Signal');

clear all; clc;

Gamma    = 9.0;         % Verstärkungsfaktor Messrauschen 
Alpha_R  = 0.5;         % Kalman-Verstärkung Varianz Messrauschen
Alpha_M  = 0.3;         % Kalman-Verstärkung Varianz M 

Ts = 0.1;               % Abtastzeit
Ad = 1;                 % Systemmatrix
Bd = 0;                 % Eingangsmatrix
C  = 1;                 % Ausgangsmatrix
D  = 0;                 % Durchgangsmatrix
G  = 1;                 % Systemfehlermatrix

%%%  M E S S D A T E N - G E N E R A T O R  %%%
t  = Ts*(1:1:1200);
ao = [zeros(1,100) +.5*ones(1,200) zeros(1,150) +1*ones(1,150)...
      zeros(1,200) -.3*ones(1,200) zeros(1,200)];
Rg = 5E-3;
v  = [sqrt(1*Rg)*randn(1,250) sqrt(10*Rg)*randn(1,350) sqrt(1*Rg)*randn(1,600)];
y  = [ao + v];
u = zeros(1,length(y));

%%%  I N I T   R O S E - F I L T E R  %%%
x_dach = [y(1)];   
p_tilde = 0; 
E1  = y(1);
EE1 = y(:,1)*y(:,1)';
M   = 0;

%%%  R O S E - F I L T E R  %%%   
for k=1:length(y)
  %--  Bestimmung R mit IIR-Filter 1. Ordnung  --
  E1  = Alpha_R*y(:,k)         + (1-Alpha_R)*E1;
  EE1 = Alpha_R*y(:,k)*y(:,k)' + (1-Alpha_R)*EE1;
  R(k)= Gamma*(EE1 - E1*E1');    

  %--  Bestimmung M mit IIR-Filter 1. Ordnung  --
  dy = y(:,k) - C*x_dach - D*u(k);     
  M  = Alpha_M.*dy*dy' + (1-Alpha_M).*M;
  
  %-- Bestimmung Q  --
  Q(k)= C'*(M - R(k))*C - Ad*p_tilde*Ad';
  if Q(k)<0 
      Q(k)=0; 
  end;  
  
  %--  Kalman Gleichungen  --
  p_dach  = Ad*p_tilde*Ad' + G*Q(k)*G';  
  K       = p_dach*C'*pinv(C*p_dach*C' + R(k)); 
  x_tilde = x_dach + K*dy;
  p_tilde = (eye(length(Bd)) - K*C)*p_dach;
  x_dach  = Ad*x_tilde + Bd*u(k);  
  x(k)    = x_tilde;
end

figure(1); 
l = length(y); 
k = [1:l];
plot(k,y,'b-',k,x,'r--'); grid on;
axis([1 l -0.8 2.3]);
xlabel('k'); 
legend('gemessenes Signal','gefiltertes Signal');

figure(2); 
subplot(211);
plot(k,R,'b-'); grid on;
axis([1 l -0.1 3]);
xlabel('k'); ylabel('R(k)');

subplot(212);
plot(k,Q,'b-'); grid on;
axis([1 l -0.05 .4]);
xlabel('k'); ylabel('Q(k)');


theta=0:pi/1024:2*pi;
L=3;y_cap=[];
rb_eq1=[];
rb_eq2=[];
rb_eq3=[];
for j=1:length(theta)
    j
   if theta(j)>=0 && theta(j) <=80*pi/180
       yv=0;
       yvd=0;yvdd=0;
       %rb_eq1=[rb_eq1;-yv-yvdd];
       y_cap=[y_cap;theta(j) yv yvd yvdd];
   end
   if theta(j)>80*pi/180 && theta(j) <=130*pi/180
       th=theta(j)-80*pi/180;beta=100*pi/180;
       yv=2*L*th^2/beta^2;
       yvd=4*L*th/beta^2;
       yvdd=4*L/beta^2;
       rb_eq1=[rb_eq1;-2*L*th^2/beta^2-4*L/beta^2];
       y_cap=[y_cap;theta(j) yv yvd yvdd];
   end
   if theta(j)>130*pi/180 && theta(j) <=180*pi/180
       th=theta(j)-80*pi/180;beta=100*pi/180;
       yv=L*(1-2*(1-(th/beta))^2);
       yvd=(4*L/beta)*(1-(th/beta));
       yvdd=-4*L/beta^2;
       rb_eq2=[rb_eq2;-L*(1-2*(1-(th/beta))^2)+4*L/beta^2];
       y_cap=[y_cap;theta(j) yv yvd yvdd];
   end
   if theta(j)>=180*pi/180 && theta(j) <=240*pi/180
       yv=3;
       yvd=0;
       yvdd=0;
       %rb_eq2=[rb_eq2;-yv-yvdd];
       y_cap=[y_cap;theta(j) yv yvd yvdd];
   end
   if theta(j)>=240*pi/180 && theta(j) <=360*pi/180
       th=theta(j)-240*pi/180;beta=120*pi/180;
       yv=(L/2)*(1+cos(pi*th/beta));
       yvd=(-pi*L/(2*beta))*sin(pi*th/beta);
       yvdd=-(L*pi^2/(2*beta^2))*cos(pi*th/beta);
       rb_eq3=[rb_eq3;-(L/2)*(1+cos(pi*th/beta))+(L*pi^2/(2*beta^2))*cos(pi*th/beta)];
       y_cap=[y_cap;theta(j) yv yvd yvdd];
   end
end   
figure(1)
plot(y_cap(:,1),y_cap(:,2),'.')
rbmin=5;
e=0.5;
thetac=[];rc=[];xcyc=[];xcyc3=[];xcyc2=[];
for j=1:length(theta)
    [j]
    xc=y_cap(j,3);
    yc=rbmin+y_cap(j,2);
    thetac=[thetac;theta(j)+atan2(xc,yc)];
    rc=[rc;sqrt(xc^2+yc^2)];     
end    
figure(2)
polar(thetac,rc)


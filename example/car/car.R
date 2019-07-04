df<-read.csv(file="car.csv", header=T)

png("car.png")

op<-par(mfrow=c(2,1),
    mar = c(3,4,1,1)+0.1)

plot(df$Measured_v_x, ylab="V_x", xlab="Time",type="l")
lines(df$Filtered_v_x,col="red",lwd=2)
legend("topright", legend=c("Measured","Filtered"), lty=1, col=c("black","red"))

plot(df$Measured_v_y, ylab="V_y", xlab="Time",type="l")
lines(df$Filtered_v_y,col="green",lwd=2)
legend("topright", legend=c("Measured","Filtered"), lty=1, col=c("black","green"))

par(op)

dev.off()
